"""
Observation Manager for the CARLA environment.
This module defines the observation space and handles observation processing.
"""

import numpy as np
from gymnasium import spaces
import carla
import math
from collections import deque

class ObservationManager:
    """
    Manages the observation space and processing for the CARLA environment.
    """
    
    def __init__(self, config=None, sensor_manager=None):
        """
        Initialize the observation manager.
        
        Args:
            config: Configuration object with observation settings
            sensor_manager: Sensor manager object
        """
        self.config = config
        self.sensor_manager = sensor_manager
        
        # Set parameters from config
        if config:
            self.num_observed_waypoints = getattr(config, 'num_observed_waypoints', 3)
            self.location_history_length = getattr(config, 'location_history_length', 10)
            self.action_history_length = getattr(config, 'action_history_length', 5)
            self.radar_resolution = getattr(config, 'radar_resolution', 3.0)
        else:
            self.num_observed_waypoints = 3
            self.location_history_length = 10
            self.action_history_length = 5
            self.radar_resolution = 3.0
        
        # Calculate number of radar points based on resolution (360 / resolution)
        self.radar_points_count = int(360 / self.radar_resolution)
        
        # Location history buffer setup
        self.location_history = deque(maxlen=self.location_history_length)
        
        # Action history buffer setup
        self.action_history = deque(maxlen=self.action_history_length)
        # Initialize with zeros
        for _ in range(self.action_history_length):
            self.action_history.append(np.zeros(2, dtype=np.float32))
        
        # Define the observation space
        self.observation_space = spaces.Dict({
            'vehicle_state': spaces.Box(
                low=np.array([-np.inf] * (8 + 2 * self.num_observed_waypoints)),  # Added 1 for trust level
                high=np.array([np.inf] * (8 + 2 * self.num_observed_waypoints)),
                dtype=np.float32
            ),
            'location_history': spaces.Box(
                low=np.array([-np.inf] * (2 * self.location_history_length)),
                high=np.array([np.inf] * (2 * self.location_history_length)),
                dtype=np.float32
            ),
            'action_history': spaces.Box(
                low=np.array([-1.0] * (2 * self.action_history_length)),
                high=np.array([1.0] * (2 * self.action_history_length)),
                dtype=np.float32
            ),
            'radar_obs': spaces.Box(
                low=0.0,
                high=20.0 if config is None else config.radar_range,  # Maximum radar range from config
                shape=(1, self.radar_points_count),  # 1 layer, resolution-based azimuth angles
                dtype=np.float32
            )
        })
    
    def update(self, vehicle, action=None):
        """
        Update observation manager state based on current vehicle state.
        This should be called every step before getting the observation.
        
        Args:
            vehicle: CARLA vehicle object
            action: Current action being applied [steering, throttle/brake]
        """
        if vehicle is None:
            return
            
        # Update location history
        location = vehicle.get_transform().location
        self.location_history.append((location.x, location.y))
        
        # Update action history if provided
        if action is not None:
            self.action_history.append(np.array(action, dtype=np.float32))
    
    def get_observation(self, vehicle, waypoints, current_waypoint_idx, waypoint_threshold, 
                       trust_interface, active_scenario, target_speed=20.0, normalized=False):
        """
        Generate observation from environment state.
        
        Args:
            vehicle: CARLA vehicle object
            waypoints: List of waypoints for path following
            current_waypoint_idx: Index of the current waypoint
            waypoint_threshold: Distance threshold to consider a waypoint reached
            trust_interface: Trust interface object
            active_scenario: Active scenario object
            target_speed: Current target speed in km/h
            
        Returns:
            obs: Observation dictionary
        """
        # Get trust level
        trust_level = trust_interface.trust_level if trust_interface else 0.75
        
        # Get vehicle state observation (now includes trust level)
        vehicle_state = self._get_vehicle_state(vehicle, waypoints, current_waypoint_idx, 
                                               waypoint_threshold, target_speed, trust_level, normalized=normalized)
        
        # Get location history
        location_history = self._get_location_history(vehicle)
        
        # Get action history
        action_history = self._get_action_history()

        # Get radar observation
        radar_observation = self.sensor_manager.get_radar_observation()
        
        # Combine all observations
        if normalized:
            obs = {
                'vehicle_state': vehicle_state,
                'location_history': location_history / 10.0,
                'action_history': action_history,
                'radar_obs': radar_observation / 20.0
            }
        else:
            obs = {
                'vehicle_state': vehicle_state,
                'location_history': location_history,
                'action_history': action_history,
                'radar_obs': radar_observation
            }
        
        return obs
    
    def _get_vehicle_state(self, vehicle, waypoints, current_waypoint_idx, waypoint_threshold, 
                          target_speed=20.0, trust_level=0.75, normalized=False):
        """Get vehicle state observation
        
        Args:
            vehicle: CARLA vehicle object
            waypoints: List of waypoints for path following
            current_waypoint_idx: Index of the current waypoint
            waypoint_threshold: Distance threshold for waypoint completion
            target_speed: Current target speed in km/h
            trust_level: Current trust level (0-1)
            
        Returns:
            numpy.ndarray: Vehicle state observation
        """
        if vehicle is None or not waypoints:
            return np.zeros(8 + 2 * self.num_observed_waypoints, dtype=np.float32)  # Updated size for trust level
            
        velocity = vehicle.get_velocity()
        angular_velocity = vehicle.get_angular_velocity()
        acceleration = vehicle.get_acceleration()
        
        if waypoints and current_waypoint_idx < len(waypoints):
            observed_waypoints = waypoints[current_waypoint_idx:current_waypoint_idx+self.num_observed_waypoints]
            
            # Pad with last waypoint if needed
            if len(observed_waypoints) < self.num_observed_waypoints and observed_waypoints:
                last_waypoint = observed_waypoints[-1]
                observed_waypoints.extend([last_waypoint] * (self.num_observed_waypoints - len(observed_waypoints)))
            
            # Initialize arrays for relative waypoint coordinates
            relative_waypoints = np.zeros((self.num_observed_waypoints, 2))
            
            # Get vehicle transform once for efficiency
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            yaw_rad = math.radians(vehicle_rotation.yaw)
            
            # Calculate relative coordinates for each observed waypoint
            for i, waypoint in enumerate(observed_waypoints):
                # Calculate relative position
                dx = waypoint.transform.location.x - vehicle_location.x
                dy = waypoint.transform.location.y - vehicle_location.y
                
                # Rotate to vehicle's coordinate system
                relative_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
                relative_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
                
                relative_waypoints[i] = np.array([relative_x, relative_y])
        else:
            # No waypoints available
            relative_waypoints = np.zeros((self.num_observed_waypoints, 2))

        # Normalize target speed (assuming max is around 50 km/h)
        normalized_target_speed = target_speed / 50.0
        if normalized:
            relative_waypoints = relative_waypoints / 20.0

        # Combine all vehicle state information
        vehicle_state = np.array([
            vehicle_transform.rotation.yaw / 360.0,
            velocity.x, velocity.y, angular_velocity.z,
            acceleration.x, acceleration.y,
            normalized_target_speed,  # Add normalized target speed
            trust_level,              # Add trust level
            *relative_waypoints.flatten()
        ], dtype=np.float32)
        
        return vehicle_state
    
    def _get_location_history(self, vehicle):
        """
        Get the vehicle's location history in a vehicle-relative coordinate system.
        
        Args:
            vehicle: CARLA vehicle object
            
        Returns:
            numpy.ndarray: Flattened array of relative past positions
        """
        if vehicle is None or len(self.location_history) == 0:
            # Return zeros if no history or no vehicle
            return np.zeros(2 * self.location_history_length, dtype=np.float32)
            
        # Get current vehicle transform
        vehicle_transform = vehicle.get_transform()
        current_location = vehicle_transform.location
        current_rotation = vehicle_transform.rotation
        yaw_rad = math.radians(current_rotation.yaw)
        
        # Initialize array for relative positions
        relative_positions = np.zeros((self.location_history_length, 2), dtype=np.float32)
        
        # Fill array with available history
        for i, (x, y) in enumerate(self.location_history):
            # Calculate relative position
            dx = x - current_location.x
            dy = y - current_location.y
            
            # Rotate to vehicle's coordinate system
            relative_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            relative_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            
            relative_positions[i] = [relative_x, relative_y]
            
        # Flatten array
        return relative_positions.flatten()
    
    def _get_action_history(self):
        """
        Get the flattened action history.
        
        Returns:
            numpy.ndarray: Flattened array of past actions
        """
        # Convert the action history deque to a flattened numpy array
        action_history_array = np.array(list(self.action_history), dtype=np.float32)
        return action_history_array.flatten()
    
    def _get_intervention_observation(self, trust_interface):
        """
        Get binary intervention observation.
        
        Args:
            trust_interface: Trust interface object
            
        Returns:
            intervention_obs: Binary value indicating recent intervention
        """
        if trust_interface:
            return trust_interface.get_intervention_observation()
        return 0
    
    def _get_scenario_observation(self, active_scenario):
        """
        Get observations related to the active scenario.
        
        Args:
            active_scenario: Active scenario object
            
        Returns:
            scenario_obs: Scenario-specific observations
        """
        # Default empty scenario observation
        scenario_obs = np.zeros(15, dtype=np.float32)
        
        if active_scenario:
            scenario_obs = active_scenario.get_observation()
        
        return scenario_obs
        
    def reset(self):
        """Reset the observation manager"""
        # Clear location history
        self.location_history.clear()
        
        # Clear action history and initialize with zeros
        self.action_history.clear()
        for _ in range(self.action_history_length):
            self.action_history.append(np.zeros(2, dtype=np.float32)) 