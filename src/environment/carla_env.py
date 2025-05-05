import carla
import gymnasium as gym
import numpy as np
import pygame
import math
import os
import json
import datetime
import cv2

from ..mdp.observation_manager import ObservationManager
from ..mdp.action_manager import ActionManager
from ..utils.env_utils import spawn_ego_vehicle, generate_random_waypoints, check_decision_points
from ..mdp.rewards import calculate_reward
from ..utils.viz_utils import render_trust_visualization, render_waypoints_on_camera
from ..utils.sensors import SensorManager
from ..utils.termination_manager import TerminationManager
from ..trust.trust_interface import TrustInterface

class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gymnasium interface"""
    
    def __init__(self, trust_interface, config, eval=False):
        self._initialized = False
        super(CarlaEnv, self).__init__()
        
        # Load configuration
        self.config = config
        self.eval = eval
        
        # Connect to CARLA server
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.get_world()
        self.render_mode = self.config.render_mode

        # Spawn new vehicle
        self.vehicle, self.spawn_point = spawn_ego_vehicle(self.world)

        # Initialize observation and action managers
        self.sensor_manager = SensorManager(self.world, self.vehicle, self.render_mode, self.config)
        self.observation_manager = ObservationManager(self.config, self.sensor_manager)
        self.action_manager = ActionManager(self.config)
        
        # Set the action and observation spaces from the managers
        self.action_space = self.action_manager.action_space
        self.observation_space = self.observation_manager.observation_space
        self.learn_starts = 0

        self.keyboard_steering = 0.0
        self.keyboard_throttle = 0.0
        
        # Scenario management
        self.active_scenario = None
        
        # Trust-related attributes
        self.trust_interface: TrustInterface = trust_interface

        # Environment attributes
        self.last_step_time = None
        self.step_count = 0
        self.max_episode_steps = self.config.max_episode_steps
        
        # Target speed attributes
        self.base_target_speed = self.config.base_target_speed  # km/h at max trust
        self.min_target_speed = self.config.min_target_speed    # km/h at min trust
        self.target_speed = self.base_target_speed  # Default to base speed initially, will be randomly set in reset()
        
        # Previous control state for detecting changes
        self.prev_control = None
        
        # Camera view setup
        self.camera_width = self.config.camera_width
        self.camera_height = self.config.camera_height
        self.pygame_initialized = False
        self.screen = None
        
        # Trust visualization
        self.trust_history = []
        self.max_trust_history = self.config.max_trust_history
        self.trust_viz_height = self.config.trust_viz_height
        
        # Reward visualization
        self.reward_history = []
        self.max_reward_history = self.config.max_reward_history
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
        
        # Path following attributes
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = self.config.waypoint_threshold  # meters
        self.path_length = self.config.path_length  # Number of waypoints to generate
        
        # Generate initial random waypoints if vehicle is spawned
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world)
        
        # Visualization settings
        self.show_waypoints = self.config.show_waypoints  # Flag to toggle waypoint visualization
        self.waypoint_lookahead = self.config.waypoint_lookahead  # Number of waypoints to show ahead
        
        # Initialize termination manager
        self.termination_manager = TerminationManager(max_episode_steps=self.max_episode_steps)

        # Recording related attributes
        self.recording = False
        self.record_track = []
        self.record_other_vehicles = []
        self.top_down_image = None
        self.spectator_camera = None
        self.record_directory = None
    
    def set_scenario(self, scenario, config=None):
        """Set the active scenario for the environment"""
        self.active_scenario = scenario
    
    def set_waypoints(self, waypoints):
        """Set the waypoints for path following"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        
    def start_recording(self, save_dir="recordings"):
        """Start recording vehicle positions and prepare top-down camera.
        
        Args:
            save_dir: Directory to save recordings
        """
        # Create save directory if it doesn't exist
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join(save_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        self.record_directory = session_dir
        
        # Clear previous recordings
        self.record_track = []
        self.record_other_vehicles = []
        
        # Set up a spectator camera above the scene
        self.setup_top_down_camera()
        
        self.recording = True
        print(f"Started recording. Data will be saved to {self.record_directory}")
    
    def _handle_input(self, action):
        """Handle keyboard input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        get_key = False
        # Steering
        if keys[pygame.K_LEFT]:
            get_key = True
            self.keyboard_steering = max(-1.0, self.keyboard_steering - 0.1)
            print(f"Left key, keyboard steering: {self.keyboard_steering}")
        elif keys[pygame.K_RIGHT]:
            get_key = True
            self.keyboard_steering = min(1.0, self.keyboard_steering + 0.1)
            print(f"Right key, keyboard steering: {self.keyboard_steering}")

        print(f"Keyboard steering: {self.keyboard_steering}")
        print(f"Keyboard throttle: {self.keyboard_throttle}")

        
        # Throttle/Brake
        if keys[pygame.K_UP]:
            get_key = True
            self.keyboard_throttle = min(1.0, self.keyboard_throttle + 0.1)
        elif keys[pygame.K_DOWN]:
            get_key = True
            self.keyboard_throttle = max(-1.0, self.keyboard_throttle - 0.1)

        if get_key:
            if isinstance(action, tuple):
                action = (self.keyboard_steering, self.keyboard_throttle)
            else:
                action[0] = self.keyboard_steering
                action[1] = self.keyboard_throttle
            
        return True
    
    def stop_recording(self):
        """Stop recording and save the data."""
        if not self.recording:
            print("No recording in progress.")
            return
        
        self.recording = False
        
        # Save recorded track data
        self.save_recorded_data()
        
        print(f"Recording stopped. Data saved to {self.record_directory}")
    
    def setup_top_down_camera(self, height=60.0):
        """Set up a top-down spectator camera.
        
        Args:
            height: Height above ground for the camera
        """
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            print("Cannot set up camera: No vehicle available")
            return
        
        # Get the current vehicle location
        vehicle_loc = self.vehicle.get_location()
        
        # Create a spectator
        spectator = self.world.get_spectator()
        
        # Position the spectator above the vehicle, looking down
        spectator_transform = carla.Transform(
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=vehicle_loc.z + height),
            carla.Rotation(pitch=-90)  # Point straight down
        )
        spectator.set_transform(spectator_transform)
        
        # Store the spectator for later use
        self.spectator_camera = spectator
    
    def update_top_down_camera(self):
        """Update the position of the top-down camera to follow the vehicle."""
        if not self.spectator_camera or not hasattr(self, 'vehicle') or self.vehicle is None:
            return
        
        # Get the current vehicle location
        vehicle_loc = self.vehicle.get_location()
        
        # Get current spectator transform
        current_transform = self.spectator_camera.get_transform()
        
        # Update only x and y to maintain height
        new_transform = carla.Transform(
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=current_transform.location.z),
            carla.Rotation(pitch=-90)  # Keep pointing straight down
        )
        
        # Apply the new transform
        self.spectator_camera.set_transform(new_transform)
    
    def capture_top_down_image(self):
        """Capture a top-down image of the scene."""
        if not self.pygame_initialized:
            print("Cannot capture image: Pygame not initialized")
            return None
        
        # Render the scene without overlays
        self.world.tick()
        
        # Use the screenshot functionality from pygame
        screenshot = pygame.Surface((self.camera_width, self.camera_height))
        screenshot.blit(self.screen, (0, 0), (0, 0, self.camera_width, self.camera_height))
        
        # Convert to numpy array
        screenshot_array = pygame.surfarray.array3d(screenshot)
        screenshot_array = np.transpose(screenshot_array, (1, 0, 2))  # Adjust dimensions
        
        # Store the image
        self.top_down_image = screenshot_array
        
        return screenshot_array
    
    def record_vehicle_positions(self):
        """Record current positions of ego vehicle and other vehicles."""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            return
        
        # Get current timestamp
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        
        # Record ego vehicle position
        ego_transform = self.vehicle.get_transform()
        ego_velocity = self.vehicle.get_velocity()
        ego_data = {
            'timestamp': timestamp,
            'x': ego_transform.location.x,
            'y': ego_transform.location.y,
            'z': ego_transform.location.z,
            'yaw': ego_transform.rotation.yaw,
            'speed': math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
        }
        self.record_track.append(ego_data)
        
        # Record other vehicles if needed
        if self.active_scenario:
            other_vehicles = []
            
            # Get all non-ego vehicles in the world
            for actor in self.world.get_actors().filter('vehicle.*'):
                if actor.id != self.vehicle.id:
                    actor_transform = actor.get_transform()
                    actor_velocity = actor.get_velocity()
                    actor_data = {
                        'id': actor.id,
                        'timestamp': timestamp,
                        'x': actor_transform.location.x,
                        'y': actor_transform.location.y,
                        'z': actor_transform.location.z,
                        'yaw': actor_transform.rotation.yaw,
                        'speed': math.sqrt(actor_velocity.x**2 + actor_velocity.y**2 + actor_velocity.z**2)
                    }
                    other_vehicles.append(actor_data)
            
            self.record_other_vehicles.append(other_vehicles)
    
    def save_recorded_data(self):
        """Save recorded data to disk."""
        if not self.record_directory:
            print("Cannot save: No recording directory set")
            return
        
        # Save track data
        track_file = os.path.join(self.record_directory, "track_data.json")
        with open(track_file, 'w') as f:
            json.dump(self.record_track, f)
        
        # Save other vehicles data if available
        if self.record_other_vehicles:
            other_vehicles_file = os.path.join(self.record_directory, "other_vehicles_data.json")
            with open(other_vehicles_file, 'w') as f:
                json.dump(self.record_other_vehicles, f)
        
        # Save waypoints
        waypoints_data = []
        if self.waypoints:
            for wp in self.waypoints:
                wp_data = {
                    'x': wp.transform.location.x,
                    'y': wp.transform.location.y,
                    'z': wp.transform.location.z,
                    'yaw': wp.transform.rotation.yaw
                }
                waypoints_data.append(wp_data)
            
            waypoints_file = os.path.join(self.record_directory, "waypoints_data.json")
            with open(waypoints_file, 'w') as f:
                json.dump(waypoints_data, f)
        
        # Save top-down image if captured
        if self.top_down_image is not None:
            image_file = os.path.join(self.record_directory, "top_down_view.png")
            # OpenCV expects BGR format for writing
            cv2.imwrite(image_file, cv2.cvtColor(self.top_down_image, cv2.COLOR_RGB2BGR))
        
        # Save some metadata
        metadata = {
            'town': self.config.town,
            'timestamp': datetime.datetime.now().isoformat(),
            'num_waypoints': len(self.waypoints) if self.waypoints else 0,
            'num_track_points': len(self.record_track),
            'num_frames_with_other_vehicles': len(self.record_other_vehicles)
        }
        
        metadata_file = os.path.join(self.record_directory, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Saved {len(self.record_track)} track points and {len(self.waypoints) if self.waypoints else 0} waypoints")
    
    def step(self, action):
        """Take a step in the environment"""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            # Return empty observation, zero reward, and terminal state if vehicle doesn't exist
            obs = self.observation_manager.get_observation(
                None, [], 0, self.waypoint_threshold, 
                self.trust_interface, self.active_scenario,
                target_speed=self.target_speed
            )
            return obs, 0.0, True, False, {}
        
        
        if isinstance(action, tuple):
            action = (action[0] * 0.5, action[1])
        else:
            action[0] *= 0.5

        self._handle_input(action)
        
        if self.step_count < self.learn_starts:
            action = self.action_space.sample()
            
        # Store previous control for comparison
        if self.prev_control is None:
            self.prev_control = self.vehicle.get_control()

        # Check for decision points (intersections, lane merges, etc.)
        is_near_decision_point = check_decision_points(self.vehicle, self.world, self.config.decision_point_distance)
        self.trust_interface.set_near_decision_point(is_near_decision_point)
            
        # Process action through action manager - with trust-based adjustments
        adjusted_action, current_intervention_prob = self.action_manager.adjust_action_with_trust(
            action, self.trust_interface, is_near_decision_point, is_adjusting=False
        )
        
        # Generate CARLA vehicle control from processed action
        control = self.action_manager.generate_vehicle_control(adjusted_action)
        
        # Apply control to vehicle
        self.vehicle.apply_control(control)
        
        # Update the world
        self.world.tick()
        
        # Update the observation manager with current vehicle state and action
        self.observation_manager.update(self.vehicle, adjusted_action)
        
        # Check if we've reached the current waypoint and update if needed
        self._update_waypoint_index()
        
        # If recording, update the top-down camera and record positions
        if self.recording:
            self.update_top_down_camera()
            self.record_vehicle_positions()
            
            # Periodically capture top-down image (every 50 steps)
            if self.step_count % 50 == 0:
                self.capture_top_down_image()
        
        # Render if needed (but don't break training if rendering fails)
        if self.render_mode:
            try:
                self.render()
            except Exception as e:
                print(f"Warning: Rendering failed: {e}")
        
        # Update driving metrics in trust interface with current target speed
        self.trust_interface.update_driving_metrics(self.vehicle, target_speed=self.target_speed)
        
        # Calculate time delta since last step
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        self.last_step_time = current_time
        
        # Detect manual interventions based on control changes and update trust
        self.trust_interface.detect_interventions_and_update_trust(
            control, 
            self.prev_control, 
            self.world.get_snapshot()
        )
        
        # Update trust-based behavior parameters in trust interface
        self._update_target_speed()
        self.trust_interface.update_behavior_adjustment()
        
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
            self.target_speed,
            reward_weights=self.config.reward_weights
        )
        
        # Update reward history for visualization
        self.episode_reward += reward
        if len(self.reward_history) >= self.max_reward_history:
            self.reward_history.pop(0)
        self.reward_history.append(reward)
        
        # Check if done
        terminated, truncated = self.termination_manager.check_termination(
            self.vehicle, 
            self.world, 
            self.step_count, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.active_scenario, 
            self.sensor_manager.collision_detected
        )
        
        # Get observation using observation manager
        obs = self.observation_manager.get_observation(
            self.vehicle, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.waypoint_threshold, 
            self.trust_interface, 
            self.active_scenario,
            target_speed=self.target_speed
        )
        
        # Store current control for next comparison
        self.prev_control = control
        
        # Increment step counter
        self.step_count += 1
        
        # Check for termination due to end of episode
        if truncated and self.recording:
            # If recording, save the data before ending the episode
            print("Episode ending, saving recording data...")
            self.capture_top_down_image()  # Capture final image
            self.stop_recording()
        
        info = {
            'current_waypoint_idx': self.current_waypoint_idx,
            'waypoints_total': len(self.waypoints),
            'progress': self.current_waypoint_idx / len(self.waypoints) if self.waypoints else 0,
            'step_count': self.step_count,
            'trust_level': self.trust_interface.trust_level,
            'target_speed': self.target_speed,
            'current_speed': math.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2) * 3.6 if self.vehicle else 0,  # Convert to km/h
            'episode_reward': self.episode_reward,
            'reward_components': self.reward_components
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
        # If recording, stop the recording before reset
        if self.recording:
            self.save_recorded_data()
            
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset step counter
        self.step_count = 0

        self.keyboard_steering = 0.0
        self.keyboard_throttle = 0.0
        
        # Reset action manager
        self.action_manager.reset()
        
        # Reset observation manager
        self.observation_manager.reset()
        
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
        
        # Reset trust-related attributes
        self.prev_control = None
        self.last_step_time = None
        self.trust_interface.reset()
        
        # Sample a random target speed for this episode
        if self.eval:
            self.target_speed = 20.0  # Fixed speed for evaluation
        else:
            self.target_speed = self.min_target_speed + np.random.random() * (self.base_target_speed - self.min_target_speed)
            print(f"Target speed for this episode: {self.target_speed:.1f} km/h")
        
        # Reset termination manager
        self.termination_manager.reset()
            
        # Destroy existing vehicle if any
        if self.vehicle.is_alive:
            self.vehicle.destroy()

        # Spawn new vehicle
        self.vehicle, self.spawn_point = spawn_ego_vehicle(self.world)
        
        # Setup sensor manager with the new vehicle
        self.sensor_manager.reset()
        self.sensor_manager.set_vehicle(self.vehicle)
        
        # Generate random waypoints for the new vehicle position
        self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world, self.config.path_length)
        
        # Setup active scenario if exists
        if self.active_scenario and not self.active_scenario.is_setup:
            self.active_scenario.setup()
        
        # Tick the world to update
        self.world.tick()
        
        # Initialize the observation manager with the vehicle's initial position and zero action
        self.observation_manager.update(self.vehicle, np.zeros(2, dtype=np.float32))
        
        # Get observation using observation manager
        obs = self.observation_manager.get_observation(
            self.vehicle, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.waypoint_threshold, 
            self.trust_interface, 
            self.active_scenario,
            target_speed=self.target_speed
        )
        
        # Additional info
        info = {
            'spawn_point': f"({self.spawn_point.location.x:.1f}, {self.spawn_point.location.y:.1f}, {self.spawn_point.location.z:.1f})",
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.75,
            'target_speed': self.target_speed,
            'fixed_target_speed': True,
            'speed_sampling': 'random' if not self.eval else 'fixed_eval',
            'speed_range': f"{self.min_target_speed}-{self.base_target_speed}" if not self.eval else "30.0"
        }
        
        # If we're recording, set up top-down camera after vehicle is spawned
        if self.recording:
            self.setup_top_down_camera()
            self.record_track = []  # Clear previous track
            self.record_other_vehicles = []
        
        return obs, info
    
    def _update_target_speed(self):
        """Update vehicle behavior parameters based on trust level and driving metrics"""
        # No longer dynamic - target speed remains fixed for the entire episode
        # It is now set randomly at reset time
        pass
    
    def close(self):
        """Clean up resources when environment is closed"""
        
        # If recording, stop and save data before closing
        if self.recording:
            self.stop_recording()
        
        # Clean up sensors
        if self.sensor_manager:
            self.sensor_manager.cleanup_sensors()
        
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

    def render(self):
        """Render the environment"""
        if self.render_mode is None or not self.pygame_initialized:
            return
            
        # Get camera image from sensor manager
        if self.sensor_manager:
            camera_array = self.sensor_manager.get_camera_image_array()
            
            if camera_array is not None:
                # Convert to pygame surface
                pygame_image = pygame.surfarray.make_surface(camera_array.swapaxes(0, 1))
                
                # Display the image
                self.screen.blit(pygame_image, (0, 0))
                
                # Draw waypoints on camera view if enabled
                if self.show_waypoints and self.waypoints and hasattr(self, 'vehicle') and self.vehicle:
                    render_waypoints_on_camera(
                        self.screen, 
                        self.sensor_manager.sensors, 
                        self.camera_width, 
                        self.camera_height, 
                        self.waypoints, 
                        self.current_waypoint_idx, self.waypoint_lookahead
                    )
                
                # Draw trust visualization panel
                if hasattr(self, 'font'):
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
                        self.max_trust_history,
                        self.target_speed
                    )
                
                # Render radar visualization
                self.sensor_manager.render_radar_visualization(
                    self.screen, 
                    self.camera_width, 
                    self.camera_height
                )
                
                # Update the display
                pygame.display.flip()
                
                # Process pygame events to keep the window responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        self.pygame_initialized = False
                
                return camera_array  # Return the RGB array for 'rgb_array' mode
        
        return None

    def _update_waypoint_index(self):
        """
        Check if we've reached the current waypoint and update the waypoint index accordingly.
        """
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints) or self.vehicle is None:
            return
            
        # Get vehicle location
        vehicle_location = self.vehicle.get_location()
        
        # Get current waypoint location
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        waypoint_location = current_waypoint.transform.location
        
        # Calculate distance to current waypoint
        distance = math.sqrt(
            (vehicle_location.x - waypoint_location.x) ** 2 +
            (vehicle_location.y - waypoint_location.y) ** 2
        )
        
        # If we've reached the waypoint, increment the index
        if distance < self.waypoint_threshold:
            self.current_waypoint_idx += 1
            
            # Log waypoint progress
            waypoints_remaining = len(self.waypoints) - self.current_waypoint_idx
            if waypoints_remaining > 0:
                print(f"Reached waypoint {self.current_waypoint_idx}/{len(self.waypoints)}. {waypoints_remaining} remaining.")
            else:
                print(f"Reached final waypoint! ({self.current_waypoint_idx}/{len(self.waypoints)})")
                
            # Limit the index to the available waypoints
            self.current_waypoint_idx = min(self.current_waypoint_idx, len(self.waypoints) - 1)

        if self.current_waypoint_idx >= len(self.waypoints) - 1:
            return
        # Get current waypoint location
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        next_waypoint = self.waypoints[self.current_waypoint_idx + 1]

        vehicle_to_current_waypoint_vector = np.array([
            current_waypoint.transform.location.x - vehicle_location.x,
            current_waypoint.transform.location.y - vehicle_location.y
        ])

        current_waypoint_to_next_waypoint_direction_vector = np.array([
            next_waypoint.transform.location.x - current_waypoint.transform.location.x,
            next_waypoint.transform.location.y - current_waypoint.transform.location.y
        ])

        
        # Normalize vectors
        vehicle_to_current_waypoint_vector = vehicle_to_current_waypoint_vector / np.linalg.norm(vehicle_to_current_waypoint_vector)
        current_waypoint_to_next_waypoint_direction_vector = current_waypoint_to_next_waypoint_direction_vector / np.linalg.norm(current_waypoint_to_next_waypoint_direction_vector)
        dot_product = np.dot(vehicle_to_current_waypoint_vector, current_waypoint_to_next_waypoint_direction_vector)
        
        angle = np.arccos(dot_product)

        if angle > math.pi / 2:
            self.current_waypoint_idx += 1
            print(f"Waypoint skipped due to large angle ({math.degrees(angle):.1f}° > {math.degrees(math.pi / 2):.1f}°)")
            
            # Limit the index to the available waypoints
            self.current_waypoint_idx = min(self.current_waypoint_idx, len(self.waypoints) - 1)