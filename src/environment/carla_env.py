import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import pygame

from ..utils.env_utils import generate_control_from_action, spawn_ego_vehicle, generate_random_waypoints, process_collision, check_decision_points
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
        self.intervention_type = None  # Track the type of intervention
        
        # Decision point detection
        self.is_near_decision_point = False
        self.decision_point_distance = 20.0  # meters
        
        # Target speed attributes
        self.base_target_speed = 20.0  # km/h at max trust
        self.min_target_speed = 5.0    # km/h at min trust
        self.target_speed = self.base_target_speed  # Default to base speed
        
        # Previous control state for detecting changes
        self.prev_control = None
        
        # Behavior adjustment parameters
        self.behavior_adjustment = {
            'trust_level': 0.5,
            'stability_factor': 1.0,
            'smoothness_factor': 1.0,
            'hesitation_factor': 1.0
        }
        
        # Sensor setup
        self.sensors = {}
        # Initialize sensor data storage
        self.collision_detected = False
        self.collision_impulse = np.zeros(3, dtype=np.float32)
        self.radar_data = None  # Store processed radar data (replacing lidar_data)
        
        # Radar configuration
        self.radar_max_distance = 20.0  # Maximum distance for radar observations (meters)
        
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
            ),
            'radar_obs': spaces.Box(
                low=0.0,
                high=self.radar_max_distance,  # Maximum radar range is now configurable
                shape=(1, 360),  # 1 layer, 360 azimuth angles (1-degree resolution)
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
            
        # Store previous control for comparison
        if self.prev_control is None:
            self.prev_control = self.vehicle.get_control()
            
        # Apply action to vehicle - with trust-based adjustments
        adjusted_action = self._adjust_action_based_on_trust(action)
        control = generate_control_from_action(adjusted_action)
        self.vehicle.apply_control(control)
        
        # Update the world
        self.world.tick()
        
        # Render if needed (but don't break training if rendering fails)
        if self.render_mode:
            try:
                self.render()
            except Exception as e:
                print(f"Warning: Rendering failed: {e}")
        
        # Check for decision points (intersections, lane merges, etc.)
        self.is_near_decision_point = check_decision_points(self.vehicle, self.world, self.decision_point_distance)
        
        # Update driving metrics in trust interface
        self.trust_interface.update_driving_metrics(self.vehicle)
        
        # Set decision point status in trust interface
        self.trust_interface.set_near_decision_point(self.is_near_decision_point)
        
        # Detect manual interventions based on control changes
        self._detect_interventions_and_update_trust(control)
        
        # Update trust-based behavior parameters
        self._update_trust_based_behavior()
        
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
        
        # Add radar observation to the observation dictionary
        obs['radar_obs'] = self.get_radar_observation()
        
        # Store current control for next comparison
        self.prev_control = control
        
        # Additional info
        info = {
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'current_speed': 3.6 * np.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2) if self.vehicle else 0.0,
            'target_speed': self.target_speed,
            'step_count': self.step_count,
            'driving_metrics': self.trust_interface.driving_metrics if self.trust_interface else {},
            'is_near_decision_point': self.is_near_decision_point,
            'behavior_adjustment': self.behavior_adjustment
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
        
        # Reset radar data with a simple structure
        self.radar_points = []  # Simple list to store current radar points
        
        # Reset trust-related attributes
        self.intervention_active = False
        self.intervention_type = None
        self.is_near_decision_point = False
        self.prev_control = None
        
        # Reset behavior adjustment
        self.behavior_adjustment = {
            'trust_level': 0.5,
            'stability_factor': 1.0,
            'smoothness_factor': 1.0,
            'hesitation_factor': 1.0
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
        
        # Add radar observation to the observation dictionary
        obs['radar_obs'] = self.get_radar_observation()
        
        # Additional info
        info = {
            'spawn_point': f"({self.spawn_point.location.x:.1f}, {self.spawn_point.location.y:.1f}, {self.spawn_point.location.z:.1f})",
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'target_speed': self.target_speed
        }
        
        return obs, info

    def _update_trust_based_behavior(self):
        """Update vehicle behavior parameters based on trust level and driving metrics"""
        # Get current trust level
        trust_level = self.trust_interface.trust_level
        
        # 1. Update target speed (as before)
        self.target_speed = self.min_target_speed + (self.base_target_speed - self.min_target_speed) * trust_level
        
        # 2. Calculate behavior adjustment factors based on trust and driving metrics
        if hasattr(self.trust_interface, 'driving_metrics'):
            # Get relevant metrics
            stability_factor = self.trust_interface.driving_metrics['steering_stability']
            smoothness_factor = (self.trust_interface.driving_metrics['acceleration_smoothness'] + 
                                self.trust_interface.driving_metrics['braking_smoothness']) / 2.0
            hesitation_factor = 1.0 - self.trust_interface.driving_metrics['hesitation_level']
            
            # Store these for potential use in action modification
            self.behavior_adjustment = {
                'trust_level': trust_level,
                'stability_factor': stability_factor,
                'smoothness_factor': smoothness_factor,
                'hesitation_factor': hesitation_factor
            }
        else:
            # Default values if metrics not available
            self.behavior_adjustment = {
                'trust_level': trust_level,
                'stability_factor': 1.0,
                'smoothness_factor': 1.0,
                'hesitation_factor': 1.0
            }
    
    def _adjust_action_based_on_trust(self, action):
        """Adjust the agent's action based on trust level and driving behavior metrics
        
        This method modifies the agent's actions to reflect how a human driver's behavior
        would change based on their trust level and driving style.
        
        Args:
            action: Original action from the agent [steering, throttle/brake]
            
        Returns:
            adjusted_action: Modified action based on trust and behavior
        """
        # Make a copy of the original action
        adjusted_action = np.array(action, dtype=np.float32)
        
        # Extract behavior factors
        
        # 1. Adjust steering based on stability
        # Low stability -> more conservative steering (reduced magnitude)
        stability_factor = self.behavior_adjustment['stability_factor']
        steering_adjustment = 0.5 + 0.5 * stability_factor  # Range: 0.5 to 1.0
        adjusted_action[0] *= steering_adjustment
        
        # 2. Adjust throttle/brake based on trust and smoothness
        # Low trust or smoothness -> more gentle acceleration, stronger braking
        trust_level = self.behavior_adjustment['trust_level']
        smoothness_factor = self.behavior_adjustment['smoothness_factor']
        if adjusted_action[1] > 0:  # Throttle
            # Low trust or smoothness -> reduce throttle
            throttle_adjustment = 0.3 + 0.7 * (trust_level * smoothness_factor)  # Range: 0.3 to 1.0
            adjusted_action[1] *= throttle_adjustment
        else:  # Brake
            # Low trust -> increase braking force
            brake_adjustment = 1.0 + (1.0 - trust_level) * 0.5  # Range: 1.0 to 1.5
            adjusted_action[1] *= brake_adjustment
        
        # 3. Add hesitation effect (random small delays or reduced actions)
        hesitation_factor = 1.0 - self.behavior_adjustment['hesitation_factor']
        if hesitation_factor > 0.3 and np.random.random() < hesitation_factor * 0.5:
            # Occasionally reduce action magnitude to simulate hesitation
            hesitation_reduction = 1.0 - (hesitation_factor * 0.5)  # Range: 0.85 to 0.5
            adjusted_action *= hesitation_reduction
        
        return adjusted_action
    
    def _detect_interventions_and_update_trust(self, current_control):
        """Detect manual interventions based on control changes"""
        if self.prev_control is None:
            return
            
        # Check for significant steering correction
        steering_change = abs(current_control.steer - self.prev_control.steer)
        if steering_change > self.trust_interface.steering_correction_threshold:
            self.intervention_active = True
            self.intervention_type = 'steer'
            self.trust_interface.update_trust(intervention=True, intervention_type='steer', dt=0.0)
            return
            
        # Check for sudden braking
        if current_control.brake > 0.7 and self.prev_control.brake < 0.3:
            self.intervention_active = True
            self.intervention_type = 'brake'
            self.trust_interface.update_trust(intervention=True, intervention_type='brake', dt=0.0)
            return

        # If no intervention detected, update trust normally
        if not self.intervention_active:
            # Calculate time delta
            current_time = self.world.get_snapshot().timestamp.elapsed_seconds
            dt = current_time - self.last_step_time if self.last_step_time is not None else 0.0
            self.last_step_time = current_time
            
            # Update trust with no intervention
            self.trust_interface.update_trust(intervention=False, dt=dt)
        else:
            # Reset intervention flag after processing
            self.intervention_active = False
            self.intervention_type = None

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
        
        # In CARLA, radar sensors don't natively support 360-degree FOV
        # We'll create multiple radar sensors to cover the full 360 degrees
        
        # Define radar parameters
        radar_range = 100.0  # Keep sensor range at 100m, but we'll filter in processing
        radar_fov = 120.0    # 120 degrees per radar (we'll use 3 radars for 360 coverage)
        
        # Create 3 radar sensors at different angles to cover 360 degrees
        radar_angles = [0, 120, 240]  # Angles in degrees
        
        for i, angle in enumerate(radar_angles):
            radar_bp = blueprint_library.find('sensor.other.radar')
            
            # Configure radar parameters
            radar_bp.set_attribute('horizontal_fov', str(radar_fov))
            radar_bp.set_attribute('vertical_fov', '20')
            radar_bp.set_attribute('range', str(radar_range))
            radar_bp.set_attribute('points_per_second', '1500')
            
            # Set the radar position and rotation
            radar_transform = carla.Transform(
                carla.Location(x=0.0, z=2.0),
                carla.Rotation(yaw=angle)  # Rotate to cover different angles
            )
            
            # Spawn the radar
            radar_sensor = self.world.spawn_actor(
                radar_bp,
                radar_transform,
                attach_to=self.vehicle
            )
            
            # Add to sensors dictionary with unique name
            self.sensors[f'radar_{i}'] = radar_sensor
            
            # Set up radar callback
            radar_sensor.listen(lambda data, radar_idx=i, radar_angle=angle: 
                               self._process_radar_data(data, radar_idx, radar_angle))
        
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
        
    def _process_radar_data(self, radar_data, radar_idx=0, radar_angle=0):
        """Process radar data with a simplified approach
        
        Args:
            radar_data: Raw radar data from CARLA
            radar_idx: Index of the radar sensor (0, 1, or 2)
            radar_angle: Base angle of the radar sensor in degrees
        """
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            return
            
        # Initialize radar_points if it doesn't exist
        if not hasattr(self, 'radar_points'):
            self.radar_points = []
            
        # Initialize radar_points_history for temporal filtering if it doesn't exist
        if not hasattr(self, 'radar_points_history'):
            self.radar_points_history = {}
            
        # When a new frame comes in from a specific radar, clear previous points from that radar
        # Filter out points from other radars
        self.radar_points = [p for p in self.radar_points if p['radar_idx'] != radar_idx]
        
        # Convert radar angle to radians
        radar_angle_rad = np.radians(radar_angle)
        
        # Temporary list to store new points from this radar update
        new_radar_points = []
        
        # Process each radar detection
        for detection in radar_data:
            # More aggressive filtering for noise reduction
            
            # 1. Filter out points that are too close or too far
            if detection.depth < 0.5 or detection.depth > self.radar_max_distance:
                continue
                
            # 2. Filter out points with low intensity (increased threshold)
            if hasattr(detection, 'intensity') and detection.intensity < 0.2:  # Increased from 0.1 to 0.2
                continue
                
            # 3. Calculate adjusted azimuth (angle)
            adjusted_azimuth = detection.azimuth + radar_angle_rad
            
            # 4. Calculate x, y coordinates (in vehicle's frame)
            x = detection.depth * np.cos(detection.altitude) * np.cos(adjusted_azimuth)
            y = detection.depth * np.cos(detection.altitude) * np.sin(adjusted_azimuth)
            
            # 5. Skip points with invalid coordinates
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                continue
                
            # 6. Filter out points with extreme altitude angles (likely ground or sky reflections)
            if abs(np.degrees(detection.altitude)) > 10:  # Filter points with altitude > 10 degrees
                continue
                
            # 7. Create a unique key for this approximate spatial location (rounded to 0.5m grid)
            # This helps track the same physical object across frames
            grid_size = 0.5  # 0.5 meter grid
            location_key = f"{int(x/grid_size)},{int(y/grid_size)}"
            
            # 8. Store essential data
            point = {
                'x': float(x),  # Ensure it's a regular float, not numpy float
                'y': float(y),
                'velocity': float(detection.velocity),
                'radar_idx': int(radar_idx),
                'location_key': location_key,
                'timestamp': self.step_count  # Track when this point was last seen
            }
            
            # Add to the temporary list of new points
            new_radar_points.append(point)
            
            # Update the history for this location
            if location_key not in self.radar_points_history:
                self.radar_points_history[location_key] = {
                    'points': [point],
                    'count': 1,
                    'last_seen': self.step_count
                }
            else:
                history = self.radar_points_history[location_key]
                history['points'].append(point)
                history['count'] += 1
                history['last_seen'] = self.step_count
                
                # Keep only the last 5 observations for this location
                if len(history['points']) > 5:
                    history['points'] = history['points'][-5:]
        
        # Apply temporal filtering - only keep points that have been seen multiple times
        # or are very recent (just detected)
        stable_points = []
        for point in new_radar_points:
            history = self.radar_points_history[point['location_key']]
            
            # Accept points that have been seen multiple times (stable)
            if history['count'] >= 3:
                # Use averaged position from history for more stability
                recent_points = history['points'][-3:]  # Last 3 observations
                avg_x = sum(p['x'] for p in recent_points) / len(recent_points)
                avg_y = sum(p['y'] for p in recent_points) / len(recent_points)
                avg_velocity = sum(p['velocity'] for p in recent_points) / len(recent_points)
                
                # Create a stabilized point
                stable_point = {
                    'x': float(avg_x),
                    'y': float(avg_y),
                    'velocity': float(avg_velocity),
                    'radar_idx': point['radar_idx'],
                    'is_stable': True  # Mark as a stable point
                }
                stable_points.append(stable_point)
            # Also accept very new points (first few detections)
            elif history['count'] <= 2:
                point['is_stable'] = False  # Mark as not yet stable
                stable_points.append(point)
        
        # Add the stable points to our main radar points list
        self.radar_points.extend(stable_points)
        
        # Clean up old history entries (not seen recently)
        keys_to_remove = []
        for key, history in self.radar_points_history.items():
            if self.step_count - history['last_seen'] > 10:  # Not seen for 10 steps
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.radar_points_history[key]
        
        # Limit the number of points to prevent memory issues
        if len(self.radar_points) > 1000:
            self.radar_points = self.radar_points[-1000:]
    
    def get_radar_observation(self):
        """Get a simplified observation from radar data
        
        Returns:
            numpy.ndarray: Array containing distance measurements at each angle
        """
        # Create a simple array to hold distance measurements
        # Shape: [360 azimuth angles] - we'll use just one layer for simplicity
        radar_obs = np.full(360, self.radar_max_distance)  # Fill with max range
        
        # If no radar data is available, return the default observation
        if not hasattr(self, 'radar_points') or not self.radar_points:
            return radar_obs.reshape(1, 360)  # Reshape to match expected dimensions
            
        # Process each point in the radar data
        for point in self.radar_points:
            # Calculate angle and distance
            x, y = point['x'], point['y']
            
            # Skip if x or y is NaN
            if np.isnan(x) or np.isnan(y):
                continue
                
            # Calculate angle in degrees (0-360)
            angle_rad = np.arctan2(y, x)
            angle_deg = np.degrees(angle_rad)
            angle_deg = (angle_deg + 360) % 360
            
            # Calculate distance
            distance = np.sqrt(x*x + y*y)
            
            # Skip points beyond our observation distance
            if distance > self.radar_max_distance:
                continue
                
            # Calculate angle index (1-degree resolution)
            angle_idx = int(angle_deg)
            if angle_idx >= 360:  # Handle edge case
                angle_idx = 359
                
            # Prioritize stable points and closer points
            if radar_obs[angle_idx] > distance or ('is_stable' in point and point['is_stable']):
                radar_obs[angle_idx] = distance
        
        # Reshape to match the expected dimensions (1 layer, 360 angles)
        return radar_obs.reshape(1, 360)
    
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
            
            # Render radar visualization (replacing LiDAR visualization)
            self._render_radar_visualization()
            
            # Update the display
            pygame.display.flip()
            
            # Process pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.pygame_initialized = False
            
            return array  # Return the RGB array for 'rgb_array' mode
        
        return None

    def _render_radar_visualization(self):
        """Render radar data visualization on the Pygame screen"""
        if not hasattr(self, 'screen') or self.screen is None:
            return
            
        # Define visualization parameters
        radar_surface_width = 200
        radar_surface_height = 200
        radar_surface = pygame.Surface((radar_surface_width, radar_surface_height))
        radar_surface.fill((0, 0, 0))  # Black background
        radar_surface.set_alpha(220)  # Slight transparency
        
        # Draw a border around the radar visualization
        pygame.draw.rect(radar_surface, (100, 100, 100), pygame.Rect(0, 0, radar_surface_width, radar_surface_height), 1)
        
        # Draw center point (ego vehicle position)
        center_x = radar_surface_width // 2
        center_y = radar_surface_height // 2
        pygame.draw.circle(radar_surface, (255, 255, 255), (center_x, center_y), 3)
        
        # Draw range circles - adjusted to show the configured max distance
        for range_val in [10, 20, self.radar_max_distance]:
            radius = int((range_val / self.radar_max_distance) * (min(center_x, center_y) - 10))
            pygame.draw.circle(radar_surface, (50, 50, 50), (center_x, center_y), radius, 1)
        
        # Draw cardinal directions - rotated so vehicle front faces north
        font = pygame.font.SysFont('Arial', 10)
        # North (vehicle front)
        text = font.render('N', True, (200, 200, 200))
        radar_surface.blit(text, (center_x - text.get_width() // 2, 5))
        # East (vehicle right)
        text = font.render('E', True, (200, 200, 200))
        radar_surface.blit(text, (radar_surface_width - 15, center_y - text.get_height() // 2))
        # South (vehicle rear)
        text = font.render('S', True, (200, 200, 200))
        radar_surface.blit(text, (center_x - text.get_width() // 2, radar_surface_height - 15))
        # West (vehicle left)
        text = font.render('W', True, (200, 200, 200))
        radar_surface.blit(text, (5, center_y - text.get_height() // 2))
        
        # Draw radar points - scale based on configured max distance
        scale_factor = (min(center_x, center_y) - 10) / self.radar_max_distance
        
        # Draw radar points
        if hasattr(self, 'radar_points') and self.radar_points:
            # Only draw points within the max distance
            visible_points = [p for p in self.radar_points if 
                             np.sqrt(p['x']**2 + p['y']**2) <= self.radar_max_distance]
            
            # Count stable vs unstable points
            stable_count = sum(1 for p in visible_points if 'is_stable' in p and p['is_stable'])
            
            for point in visible_points:
                # Skip points with NaN values
                if np.isnan(point['x']) or np.isnan(point['y']):
                    continue
                    
                # Rotate coordinates so vehicle front faces north
                # In the original coordinate system: x is forward, y is right
                # In the rotated system: x is right, y is backward (to match screen coordinates)
                rotated_x = point['y']  # Original y becomes x (right)
                rotated_y = -point['x']  # Negative original x becomes y (down)
                
                # Convert to display coordinates
                x = center_x + int(rotated_x * scale_factor)
                y = center_y + int(rotated_y * scale_factor)
                
                # Check if point is within display bounds
                if 0 <= x < radar_surface_width and 0 <= y < radar_surface_height:
                    # All points are green, but stable points are brighter
                    if 'is_stable' in point and point['is_stable']:
                        color = (0, 255, 0)  # Bright green for stable points
                    else:
                        color = (0, 150, 0)  # Darker green for unstable points
                    
                    # Size still varies with velocity for approaching objects
                    velocity = point['velocity']
                    point_size = 2
                    if velocity < 0:  # Approaching
                        point_size = min(5, 2 + int(abs(velocity) / 5))
                    
                    pygame.draw.circle(radar_surface, color, (x, y), point_size)
            
            # Add point count to the visualization
            count_text = font.render(f"Points: {len(visible_points)} (Stable: {stable_count})", True, (200, 200, 200))
            radar_surface.blit(count_text, (5, radar_surface_height - 20))
        
        # Add a title with max distance and resolution
        title_font = pygame.font.SysFont('Arial', 12)
        title_text = title_font.render(f'Radar View ({self.radar_max_distance}m, 1° res)', True, (255, 255, 255))
        radar_surface.blit(title_text, (center_x - title_text.get_width() // 2, 5))
        
        # Position the radar visualization in the bottom-right corner of the camera view
        self.screen.blit(radar_surface, (self.camera_width - radar_surface_width - 10, 
                                         self.camera_height - radar_surface_height - 10))
