import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gymnasium interface"""
    
    def __init__(self, town='Town01', port=2000, trust_interface = None):
        self._initialized = False
        super(CarlaEnv, self).__init__()
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Scenario management
        self.active_scenario = None
        self.scenario_config = None
        
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
        self._setup_sensors()
        
        # Set up action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # [steering, throttle/brake]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space includes vehicle state, path info, and intervention history
        self.observation_space = spaces.Dict({
            'vehicle_state': spaces.Box(
                low=np.array([-np.inf] * 12),
                high=np.array([np.inf] * 12),
                dtype=np.float32
            ),  # [speed_x, speed_y, accel_x, accel_y, angular_velocity, steering, throttle, brake, 
                #  distance_to_waypoint, angle_to_waypoint, next_waypoint_x, next_waypoint_y]
            'recent_intervention': spaces.Discrete(2),  # Binary: 0 or 1
            'scenario_obs': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20,),  # Adjust size based on scenario needs
                dtype=np.float32
            )
        })
        
        # Path following attributes
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 2.0  # meters
        
    def set_scenario(self, scenario, config=None):
        """Set the active scenario for the environment"""
        self.active_scenario = scenario
        self.scenario_config = config
    
    def set_waypoints(self, waypoints):
        """Set the waypoints for path following"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
    
    def step(self, action):
        """Take a step in the environment"""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            return self._get_obs(), 0.0, True, False, {}
            
        # Apply action to vehicle
        control = carla.VehicleControl()
        control.steer = float(np.clip(action[0], -1.0, 1.0))
        
        # Throttle and brake are combined in the second action value
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        if throttle_brake >= 0.0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
            
        self.vehicle.apply_control(control)
        
        # Update the world
        self.world.tick()
        
        # Update trust-based target speed if trust interface is available
        if self.trust_interface:
            self._update_trust_based_speed()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        terminated = self._is_done()
        truncated = False  # We don't use truncation in this environment
        
        # Get observation
        obs = self._get_obs()
        
        # Additional info
        info = {
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'current_speed': 3.6 * np.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2) if self.vehicle else 0.0,
            'target_speed': self.target_speed
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
            
        # Destroy existing vehicle if any
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            # First destroy sensors
            if hasattr(self, 'sensors'):
                for sensor in self.sensors.values():
                    if sensor.is_alive:
                        sensor.destroy()
                self.sensors = {}
            
            # Then destroy vehicle
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
        
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Find a valid spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise ValueError("No spawn points available in the map")
            
        # Randomly select a spawn point
        spawn_point = np.random.choice(spawn_points)
        
        # Try to spawn the vehicle
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        
        # If spawning failed (e.g., collision with existing object), try other spawn points
        if self.vehicle is None:
            # Try a few random spawn points
            for _ in range(10):
                spawn_point = np.random.choice(spawn_points)
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is not None:
                    break
            
            # If still failed, try all spawn points sequentially
            if self.vehicle is None:
                for spawn_point in spawn_points:
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if self.vehicle is not None:
                        break
            
            # If all spawn points failed, raise an error
            if self.vehicle is None:
                raise RuntimeError("Failed to spawn vehicle at any spawn point")
        
        # Setup sensors
        self._setup_sensors()
        
        # Reset waypoint tracking
        self.current_waypoint_idx = 0
        
        # Setup active scenario if exists
        if self.active_scenario:
            self.active_scenario.setup()
        
        # Tick the world to update
        self.world.tick()
        
        # Get initial observation
        obs = self._get_obs()
        
        # Additional info
        info = {
            'spawn_point': f"({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}, {spawn_point.location.z:.1f})",
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'target_speed': self.target_speed
        }
        
        return obs, info
    
    def _get_obs(self):
        """Get current observation of the environment"""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            return {
                'vehicle_state': np.zeros(12),
                'recent_intervention': 0,
                'scenario_obs': np.zeros(20)
            }
        
        # Get vehicle state
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        angular_velocity = self.vehicle.get_angular_velocity()
        control = self.vehicle.get_control()
        
        # Get path following info
        distance_to_waypoint = float('inf')
        angle_to_waypoint = 0.0
        next_waypoint_x = 0.0
        next_waypoint_y = 0.0
        
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            ego_transform = self.vehicle.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            
            # Get next waypoint
            next_waypoint = self.waypoints[self.current_waypoint_idx]
            next_waypoint_x = next_waypoint.x
            next_waypoint_y = next_waypoint.y
            
            # Calculate distance to waypoint
            distance_to_waypoint = np.sqrt(
                (ego_location.x - next_waypoint.x) ** 2 +
                (ego_location.y - next_waypoint.y) ** 2
            )
            
            # Calculate angle to waypoint
            waypoint_vector = carla.Vector3D(
                x=next_waypoint.x - ego_location.x,
                y=next_waypoint.y - ego_location.y,
                z=0.0
            )
            
            # Calculate angle between forward vector and waypoint vector
            dot = ego_forward.x * waypoint_vector.x + ego_forward.y * waypoint_vector.y
            cross = ego_forward.x * waypoint_vector.y - ego_forward.y * waypoint_vector.x
            angle_to_waypoint = np.arctan2(cross, dot)
            
            # Update waypoint index if close enough
            if distance_to_waypoint < self.waypoint_threshold:
                self.current_waypoint_idx += 1
        
        vehicle_state = np.array([
            velocity.x, velocity.y,              # Linear velocity
            acceleration.x, acceleration.y,       # Linear acceleration
            angular_velocity.z,                   # Angular velocity (yaw rate)
            control.steer,                       # Current steering
            control.throttle,                    # Current throttle
            control.brake,                       # Current brake
            distance_to_waypoint,                # Distance to next waypoint
            angle_to_waypoint,                   # Angle to next waypoint
            next_waypoint_x,                     # Next waypoint x coordinate
            next_waypoint_y                      # Next waypoint y coordinate
        ])
        
        # Get intervention state
        recent_intervention = (
            self.trust_interface.get_intervention_observation()
            if self.trust_interface is not None else 0
        )
        
        # Get scenario-specific observations
        if self.active_scenario:
            scenario_obs = self.active_scenario.get_scenario_specific_obs()
        else:
            scenario_obs = np.zeros(20)
        
        scenario_obs = np.pad(
            scenario_obs,
            (0, 20 - len(scenario_obs)),
            'constant',
            constant_values=0
        )
        
        return {
            'vehicle_state': vehicle_state,
            'recent_intervention': recent_intervention,
            'scenario_obs': scenario_obs
        }

    def _update_trust_based_speed(self):
        """Calculate target speed based on trust level"""
        if self.trust_interface:
            # Linear interpolation between min and max speed based on trust
            trust_level = self.trust_interface.trust_level
            self.target_speed = self.min_target_speed + (self.base_target_speed - self.min_target_speed) * trust_level

    
    def _calculate_reward(self):
        """Calculate reward based on current state"""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            return 0.0
            
        # Get current vehicle state
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # km/h
        acceleration = self.vehicle.get_acceleration()
        current_accel = np.sqrt(acceleration.x**2 + acceleration.y**2)
        
        # Path following reward
        path_reward = 0.0
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            ego_transform = self.vehicle.get_transform()
            ego_location = ego_transform.location
            next_waypoint = self.waypoints[self.current_waypoint_idx]
            
            # Distance to waypoint
            distance = np.sqrt(
                (ego_location.x - next_waypoint.x) ** 2 +
                (ego_location.y - next_waypoint.y) ** 2
            )
            
            # Reward for being close to waypoint
            path_reward = 1.0 - min(1.0, distance / 10.0)  # Max distance of 10 meters
            
            # Additional reward for reaching waypoint
            if distance < self.waypoint_threshold:
                path_reward += 2.0
        
        # Progress reward (based on speed)
        # Use trust-based target speed instead of fixed value
        speed_diff = abs(current_speed - self.target_speed)
        progress_reward = 1.0 - min(1.0, speed_diff / max(1.0, self.target_speed))  # Avoid division by zero
        
        # Safety reward components
        safety_reward = 0.0
        if self.active_scenario:
            danger_threshold = 5.0  # meters
            min_distance = float('inf')
            ego_location = self.vehicle.get_location()
            
            vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                if vehicle.id != self.vehicle.id:
                    distance = ego_location.distance(vehicle.get_location())
                    min_distance = min(min_distance, distance)
            
            if min_distance < danger_threshold:
                safety_reward = -1.0 * (1.0 - min_distance / danger_threshold)
        
        # Comfort reward (penalize high acceleration and jerk)
        max_comfortable_accel = 3.0  # m/s²
        comfort_reward = -min(1.0, current_accel / max_comfortable_accel)
        
        # Trust-based reward
        trust_reward = self.trust_interface.trust_level if self.trust_interface else 0.5
        
        # Intervention penalty
        intervention_penalty = -2.0 if (self.trust_interface and self.trust_interface.intervention_active) else 0.0
        
        # Combine rewards with weights
        total_reward = (
            0.4 * path_reward +        # Weight for path following
            0.2 * progress_reward +    # Weight for maintaining target speed
            0.2 * safety_reward +      # Weight for safety distance
            0.1 * comfort_reward +     # Weight for smooth driving
            0.1 * trust_reward +       # Weight for trust level
            intervention_penalty       # Full penalty for interventions
        )
        
        return total_reward
    
    def _is_done(self):
        """Check if episode is terminated
        
        Returns:
            bool: True if the episode should be terminated, False otherwise
        """
            
        if self.active_scenario and self.active_scenario.check_scenario_completion():
            return True
            
        return False
    
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
        if self.trust_interface:
            self.trust_interface.cleanup()