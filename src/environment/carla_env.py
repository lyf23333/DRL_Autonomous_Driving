import carla
import gym
import numpy as np
from gym import spaces


class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gym interface"""
    
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
        """Execute one time step within the environment"""
        # Get current time for trust updates
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        
        # Calculate dt for trust updates
        if self.last_step_time is not None:
            dt = current_time - self.last_step_time
        else:
            dt = 0.0
        self.last_step_time = current_time
        
        # Get current speed for trust-based intervention
        current_speed = 0.0
        if self.vehicle is not None:
            velocity = self.vehicle.get_velocity()
            current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # Convert to km/h
        
        # Check for trust-based intervention
        if self.trust_interface is not None and self._initialized:
            should_intervene = self.trust_interface.should_intervene(current_time, current_speed)
            if should_intervene:
                # Override action with emergency brake
                action = np.array([0.0, -1.0])  # No steering, full brake
            
            # Update trust level based on intervention and action smoothness
            self.trust_interface.update_trust(
                intervention=should_intervene,
                dt=dt
            )
        
        # Apply action
        control = carla.VehicleControl(
            throttle=float(action[1]) if action[1] > 0 else 0,
            brake=float(-action[1]) if action[1] < 0 else 0,
            steer=float(action[0])
        )
        self.vehicle.apply_control(control)
        
        # Tick the simulation
        self.world.tick()
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'intervention_active': self.trust_interface.intervention_active if self.trust_interface else False,
            'recent_interventions': self.trust_interface.get_recent_interventions() if self.trust_interface else 0,
            'current_speed': current_speed
        }
        
        # Add scenario-specific info
        if self.active_scenario:
            info['scenario_complete'] = self.active_scenario.check_scenario_completion()

        
        return obs, reward, done, info

    def reset(self):
        """Reset the environment"""
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
        
        return self._get_obs()
    
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
        target_speed = 20.0  # km/h
        speed_diff = abs(current_speed - target_speed)
        progress_reward = 1.0 - min(1.0, speed_diff / target_speed)
        
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
        """Check if episode is done"""
            
        if self.active_scenario and self.active_scenario.check_scenario_completion():
            return True
            
        return False
    
    def close(self):
        """Cleanup the environment"""
        if self.active_scenario:
            self.active_scenario.cleanup()
        
        if hasattr(self, 'vehicle'):
            self.vehicle.destroy() 

        if self.trust_interface:
            self.trust_interface.cleanup()