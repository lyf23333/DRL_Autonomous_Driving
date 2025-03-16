import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import pygame

from ..utils.env_utils import generate_control_from_action, spawn_ego_vehicle, generate_random_waypoints, process_collision
from ..mdp.observation import get_obs
from ..mdp.rewards import calculate_reward
from ..utils.viz_utils import render_trust_visualization, render_waypoints_on_camera

class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gymnasium interface"""
    
    def __init__(self, trust_interface, town='Town01', port=2000, render_mode=None):
        self._initialized = False
        super(CarlaEnv, self).__init__()
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.step_count = 0
        self.max_episode_steps = 1000
        
        # Scenario management
        self.active_scenario = None
        
        # Trust-related attributes
        self.trust_interface = trust_interface
        self.last_step_time = None
        self.intervention_active = False
        
        # Target speed attributes
        self.base_target_speed = 20.0  # km/h at max trust
        self.min_target_speed = 5.0    # km/h at min trust
        self.target_speed = self.base_target_speed  # Default to base speed
        
        # Sensor setup
        self.sensors = {}
        # Initialize sensor data storage
        self.collision_detected = False
        self.collision_impulse = np.zeros(3, dtype=np.float32)
        
        # Camera view setup
        self.camera_width = 800
        self.camera_height = 600
        self.camera_image = None
        self.render_mode = render_mode  # Can be None, 'human', or 'rgb_array'
        self.pygame_initialized = False
        self.screen = None
        
        # Trust visualization
        self.trust_history = []
        self.max_trust_history = 100  # Number of trust values to keep in history
        self.trust_viz_height = 220  # Increased from 120 to 180 for more space
        
        # Reward visualization
        self.reward_history = []
        self.max_reward_history = 100  # Number of reward values to keep in history
        self.episode_reward = 0.0      # Cumulative reward for current episode
        
        # Reward component tracking
        self.reward_components = {
            'path': 0.0,
            'progress': 0.0,
            'safety': 0.0,
            'comfort': 0.0,
            'trust': 0.0,
            'intervention': 0.0
        }
        
        # Only initialize pygame if render_mode is True
        if self.render_mode:
            try:
                pygame.init()
                # Create a taller screen to accommodate trust visualization
                self.screen = pygame.display.set_mode((self.camera_width, self.camera_height + self.trust_viz_height))
                pygame.display.set_caption("CARLA Environment - Vehicle View")
                self.pygame_initialized = True
                # Initialize font for text rendering
                self.font = pygame.font.SysFont('Arial', 16)
            except:
                print("Warning: Pygame initialization failed. Camera view will not be displayed.")
                self.pygame_initialized = False
        
        # Set up action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # [steering, throttle/brake]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space includes vehicle state, path info, and intervention history
        self.observation_space = spaces.Dict({
            'vehicle_state': spaces.Box(
                low=np.array([-np.inf] * 16),
                high=np.array([np.inf] * 16),
                dtype=np.float32
            ),
            'recent_intervention': spaces.Discrete(2),  # Binary: 0 or 1
            'scenario_obs': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(15,),  # Updated for 3 vehicles with 5 values each
                dtype=np.float32
            )
        })

        # Path following attributes
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 2.0  # meters
        self.path_length = 20  # Number of waypoints to generate
        
        # Generate initial random waypoints if vehicle is spawned
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world)
        
        # Visualization settings
        self.show_waypoints = True  # Flag to toggle waypoint visualization
        self.waypoint_lookahead = 20  # Number of waypoints to show ahead
        
    def set_scenario(self, scenario, config=None):
        """Set the active scenario for the environment"""
        self.active_scenario = scenario
    
    def set_waypoints(self, waypoints):
        """Set the waypoints for path following"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
    
    def step(self, action):
        """Take a step in the environment"""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            return self._get_obs(), 0.0, True, False, {}
            
        # Apply action to vehicle
        control = generate_control_from_action(action)
        self.vehicle.apply_control(control)
        
        # Update the world
        self.world.tick()
        
        # Render if needed (but don't break training if rendering fails)
        if self.render_mode:
            try:
                self.render()
            except Exception as e:
                print(f"Warning: Rendering failed: {e}")
        
        # Update trust-based target speed if trust interface is available
        self._update_trust_based_speed()
        # Update trust history for visualization
        if len(self.trust_history) >= self.max_trust_history:
            self.trust_history.pop(0)
        self.trust_history.append(self.trust_interface.trust_level)
        
        # Calculate reward
        reward, self.reward_components = calculate_reward(
            self.vehicle, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.waypoint_threshold, 
            self.trust_interface, 
            self.active_scenario, 
            self.world, 
            self.target_speed
        )
        
        # Update reward history for visualization
        self.episode_reward += reward
        if len(self.reward_history) >= self.max_reward_history:
            self.reward_history.pop(0)
        self.reward_history.append(reward)
        
        self.step_count += 1
        
        # Check if done
        terminated, truncated = self._is_done()
        
        # Get observation
        obs = get_obs(self.vehicle, self.waypoints, self.current_waypoint_idx, self.waypoint_threshold, self.trust_interface, self.active_scenario)
        
        # Additional info
        info = {
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'current_speed': 3.6 * np.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2) if self.vehicle else 0.0,
            'target_speed': self.target_speed,
            'step_count': self.step_count
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reset the environment
        
        Args:
            seed: The seed for random number generation
            options: Additional options for environment reset
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset sensor data
        self.collision_detected = False
        self.collision_impulse = np.zeros(3, dtype=np.float32)
        
        # Reset step counter
        self.step_count = 0
        
        # Reset stuck detection
        self.low_speed_counter = 0
        
        # Reset trust history
        self.trust_history = []
        
        # Reset reward history
        self.reward_history = []
        self.episode_reward = 0.0
        
        # Reset reward components
        self.reward_components = {
            'path': 0.0,
            'progress': 0.0,
            'safety': 0.0,
            'comfort': 0.0,
            'trust': 0.0,
            'intervention': 0.0
        }
        
        # Destroy existing vehicle if any
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            # First destroy sensors
            if hasattr(self, 'sensors'):
                for sensor in self.sensors.values():
                    if sensor and sensor.is_alive:
                        sensor.destroy()
                self.sensors = {}
            
            # Then destroy vehicle
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
        
        self.vehicle, self.spawn_point = spawn_ego_vehicle(self.world)
        
        # Setup sensors
        self._setup_sensors()
        
        # Generate random waypoints for the new vehicle position
        self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world)

        # Setup active scenario if exists
        if self.active_scenario and not self.active_scenario.is_setup:
            self.active_scenario.setup()

        # Tick the world to update
        self.world.tick()

        # Get initial observation
        obs = get_obs(self.vehicle, self.waypoints, self.current_waypoint_idx, self.waypoint_threshold, self.trust_interface, self.active_scenario)

        # Additional info
        info = {
            'spawn_point': f"({self.spawn_point.location.x:.1f}, {self.spawn_point.location.y:.1f}, {self.spawn_point.location.z:.1f})",
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'target_speed': self.target_speed
        }
        
        return obs, info

    def _update_trust_based_speed(self):
        """Calculate target speed based on trust level"""
        # Linear interpolation between min and max speed based on trust
        trust_level = self.trust_interface.trust_level
        self.target_speed = self.min_target_speed + (self.base_target_speed - self.min_target_speed) * trust_level


    def _is_done(self):
        """Check if episode is terminated or truncated
        
        In reinforcement learning environments:
        - Termination: Natural end of an episode due to failure or impossible recovery.
          The agent receives no further rewards after termination.
        - Truncation: Artificial end of an episode due to time limits or successful completion.
          The episode could have continued, but we choose to end it.
        
        Termination conditions (failure states):
        - Vehicle doesn't exist
        - Vehicle is off-road
        - Vehicle is stuck (very low speed for extended time)
        - Collision detected (if configured to terminate on collision)
        
        Truncation conditions (success or time limit):
        - Scenario is completed successfully
        - Vehicle reached the end of the path
        - Maximum number of steps reached
        
        Returns:
            terminated: True if the episode should be terminated (failure state)
            truncated: True if the episode should be truncated (success or time limit)
        """
        # Initialize return values
        terminated = False
        truncated = False
        
        # Check if vehicle exists
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            terminated = True
            
        # Check if active scenario is completed
        if self.active_scenario and self.active_scenario.check_scenario_completion():
            truncated = True
            
        # Check if vehicle reached the end of the path - TRUNCATE
        if self.waypoints and self.current_waypoint_idx >= len(self.waypoints):
            print("Episode truncated: Reached end of path")
            truncated = True
            
        # Check if vehicle is off-road - TERMINATE
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        if current_waypoint is None:
            print("Episode terminated: Vehicle is off-road")
            terminated = True
            
        # Check if vehicle is stuck (very low speed for extended time) - TERMINATE
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # km/h
        
        if not hasattr(self, 'low_speed_counter'):
            self.low_speed_counter = 0
            
        if current_speed < 1.0:  # Less than 1 km/h
            self.low_speed_counter += 1
        else:
            self.low_speed_counter = 0
            
        if self.low_speed_counter > 100:  # Stuck for too long
            print("Episode terminated: Vehicle is stuck")
            terminated = True
            
        # Check if maximum episode length reached - TRUNCATE
        
        if self.step_count >= self.max_episode_steps:
            print(f"Episode truncated: Reached maximum episode length ({self.max_episode_steps} steps)")
            truncated = True

        if self.collision_detected:
            terminated = True
            
        return terminated, truncated
    
    def close(self):
        """Clean up resources when environment is closed"""
        
        # Clean up sensors
        if hasattr(self, 'sensors'):
            for sensor in self.sensors.values():
                if sensor and sensor.is_alive:
                    sensor.destroy()
            self.sensors = {}
        
        # Clean up vehicle
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
        
        # Clean up scenario
        if self.active_scenario:
            self.active_scenario.cleanup()

        # Clean up trust interface
        self.trust_interface.cleanup()
            
        # Clean up pygame
        if hasattr(self, 'pygame_initialized') and self.pygame_initialized:
            try:
                pygame.quit()
                self.pygame_initialized = False
            except:
                pass

    def _setup_sensors(self):
        """Setup sensors for the vehicle"""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            return
            
        # Clean up existing sensors if any
        for sensor in self.sensors.values():
            if sensor.is_alive:
                sensor.destroy()
        self.sensors = {}
        
        blueprint_library = self.world.get_blueprint_library()
        
        # Collision sensor setup
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.sensors['collision'] = collision_sensor
        
        # Set up collision sensor callback
        collision_sensor.listen(lambda event: self._process_collision(event))
        
        # Only set up camera if rendering is enabled
        if self.render_mode:
            # Camera setup for visualization
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.camera_width))
            camera_bp.set_attribute('image_size_y', str(self.camera_height))
            camera_bp.set_attribute('fov', '90')
            
            # Set the camera position relative to the vehicle
            camera_transform = carla.Transform(
                carla.Location(x=1.6, z=1.7),  # Position slightly above and forward of the hood
                carla.Rotation(pitch=-15)       # Angle slightly downward
            )
            
            # Spawn the camera
            camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.vehicle
            )
            self.sensors['camera'] = camera
            
            # Set up camera callback
            camera.listen(lambda image: self._process_camera_data(image))
    
    def _process_collision(self, event):
        """Process collision events"""
        # Use the utility function to process the collision
        self.collision_detected, self.collision_impulse = process_collision(event, self)
        
    def _process_camera_data(self, image):
        """Process camera data from sensor"""
        self.camera_image = image
        
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
            
        if not self.pygame_initialized:
            self._init_pygame()
            
        if self.camera_image is not None:
            # Convert CARLA image to numpy array properly
            array = np.frombuffer(self.camera_image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.camera_image.height, self.camera_image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            array = array[:, :, ::-1]  # Convert from BGR to RGB
            
            # Convert to pygame surface
            pygame_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
            # Display the image
            self.screen.blit(pygame_image, (0, 0))
            
            # Draw waypoints on camera view if enabled
            if self.show_waypoints and self.waypoints and hasattr(self, 'vehicle') and self.vehicle:
                render_waypoints_on_camera(
                    self.screen, 
                    self.sensors, 
                    self.camera_width, 
                    self.camera_height, 
                    self.waypoints, 
                    self.current_waypoint_idx, self.waypoint_lookahead
                )
            
            # Draw trust visualization panel
            if self.pygame_initialized and hasattr(self, 'font'):
                render_trust_visualization(
                    self.screen, 
                    self.font, 
                    self.trust_interface, 
                    self.vehicle, 
                    self.camera_width, 
                    self.camera_height, 
                    self.trust_viz_height, 
                    self.reward_components, 
                    self.trust_history, 
                    self.max_trust_history
                )
            
            # Update the display
            pygame.display.flip()
            
            # Process pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.pygame_initialized = False
            
            return array  # Return the RGB array for 'rgb_array' mode
        
        return None
