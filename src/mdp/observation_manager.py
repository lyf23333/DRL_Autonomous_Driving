"""
Observation Manager for the CARLA environment.
This module defines the observation space and handles observation processing.
"""

import numpy as np
from gymnasium import spaces
import carla
import math

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
        self.num_observed_waypoints = 3
        # Define the observation space
        self.observation_space = spaces.Dict({
            'vehicle_state': spaces.Box(
                low=np.array([-np.inf] * (9 + 2 * self.num_observed_waypoints)),
                high=np.array([np.inf] * (9 + 2 * self.num_observed_waypoints)),
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
                high=20.0,  # Maximum radar range is configurable
                shape=(1, 360),  # 1 layer, 360 azimuth angles (1-degree resolution)
                dtype=np.float32
            )
        })
    
    def get_observation(self, vehicle, waypoints, current_waypoint_idx, waypoint_threshold, 
                       trust_interface, active_scenario):
        """
        Generate observation from environment state.
        
        Args:
            vehicle: CARLA vehicle object
            waypoints: List of waypoints for path following
            current_waypoint_idx: Index of the current waypoint
            waypoint_threshold: Distance threshold to consider a waypoint reached
            trust_interface: Trust interface object
            active_scenario: Active scenario object
            radar_observation: Radar observation data
            
        Returns:
            obs: Observation dictionary
        """
        # Get vehicle state observation
        vehicle_state = self._get_vehicle_state(vehicle, waypoints, current_waypoint_idx, waypoint_threshold)
        
        # Get intervention observation
        intervention_obs = self._get_intervention_observation(trust_interface)
        
        # Get scenario observation
        scenario_obs = self._get_scenario_observation(active_scenario)

        # Get radar observation
        radar_observation = self.sensor_manager.get_radar_observation()
        
        # Combine all observations
        obs = {
            'vehicle_state': vehicle_state,
            'recent_intervention': intervention_obs,
            'scenario_obs': scenario_obs,
            'radar_obs': radar_observation
        }
        
        return obs
    
    def _get_vehicle_state(self, vehicle, waypoints, current_waypoint_idx, waypoint_threshold):
        """Get vehicle state observation
        
        Args:
            vehicle: CARLA vehicle object
            waypoints: List of waypoints for path following
            current_waypoint_idx: Index of the current waypoint
            waypoint_threshold: Distance threshold for waypoint completion
            
        Returns:
            numpy.ndarray: Vehicle state observation
        """
        velocity = vehicle.get_velocity()
        angular_velocity = vehicle.get_angular_velocity()
        acceleration = vehicle.get_acceleration()
        
        if waypoints and current_waypoint_idx < len(waypoints):
            observed_waypoints = waypoints[current_waypoint_idx:current_waypoint_idx+self.num_observed_waypoints]
        
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

        # Combine all vehicle state information
        vehicle_state = np.array([
            velocity.x, velocity.y, velocity.z,
            angular_velocity.x, angular_velocity.y, angular_velocity.z,
            acceleration.x, acceleration.y, acceleration.z,
            *relative_waypoints.flatten()
        ], dtype=np.float32)
        
        return vehicle_state
    
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
            vehicle: CARLA vehicle object
            
        Returns:
            scenario_obs: Scenario-specific observations
        """
        # Default empty scenario observation
        scenario_obs = np.zeros(15, dtype=np.float32)
        
        if active_scenario:
            scenario_obs = active_scenario.get_observation()
        
        return scenario_obs 