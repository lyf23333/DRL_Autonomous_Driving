import numpy as np
import carla
import math


def get_obs(vehicle, waypoints, current_waypoint_idx, waypoint_threshold, trust_interface, active_scenario):
    """Get observation for the RL agent
    
    Args:
        vehicle: CARLA vehicle object
        waypoints: List of waypoints for path following
        current_waypoint_idx: Index of the current waypoint
        waypoint_threshold: Distance threshold for waypoint completion
        trust_interface: Trust interface object
        active_scenario: Active scenario object
        
    Returns:
        dict: Observation dictionary
    """
    # Default observation if vehicle doesn't exist
    if vehicle is None:
        return {
            'vehicle_state': np.zeros(16, dtype=np.float32),
            'recent_intervention': 0,
            'scenario_obs': np.zeros(15, dtype=np.float32),
            'radar_obs': np.zeros((1, 360), dtype=np.float32)  # Updated to match SensorManager's radar shape
        }
    
    # Get vehicle state
    vehicle_state = get_vehicle_state(vehicle, waypoints, current_waypoint_idx, waypoint_threshold)
    
    # Get recent intervention
    recent_intervention = (
        trust_interface.get_intervention_observation()
        if trust_interface is not None else 0
    )
    
    # Get scenario-specific observation
    scenario_obs = get_scenario_observation(vehicle, active_scenario)
    
    # Combine observations
    obs = {
        'vehicle_state': vehicle_state,
        'recent_intervention': recent_intervention,
        'scenario_obs': scenario_obs,
        # Note: radar_obs is added by the environment after calling this function
    }
    
    return obs

def get_vehicle_state(vehicle, waypoints, current_waypoint_idx, waypoint_threshold):
    """Get vehicle state observation
    
    Args:
        vehicle: CARLA vehicle object
        waypoints: List of waypoints for path following
        current_waypoint_idx: Index of the current waypoint
        waypoint_threshold: Distance threshold for waypoint completion
        
    Returns:
        numpy.ndarray: Vehicle state observation
    """
    # Get vehicle transform
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    
    # Get vehicle velocity and angular velocity
    velocity = vehicle.get_velocity()
    angular_velocity = vehicle.get_angular_velocity()
    
    # Get vehicle acceleration
    acceleration = vehicle.get_acceleration()
    
    # Calculate speed (in km/h)
    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    # Get current waypoint and next waypoint
    current_waypoint = None
    next_waypoint = None
    
    if waypoints and current_waypoint_idx < len(waypoints):
        current_waypoint = waypoints[current_waypoint_idx]
        
        # Calculate distance to current waypoint
        distance_to_waypoint = math.sqrt(
            (location.x - current_waypoint.transform.location.x)**2 +
            (location.y - current_waypoint.transform.location.y)**2
        )
        
        # Check if we need to move to the next waypoint
        if distance_to_waypoint < waypoint_threshold and current_waypoint_idx + 1 < len(waypoints):
            next_waypoint = waypoints[current_waypoint_idx + 1]
        else:
            next_waypoint = current_waypoint
    
    # Default values if waypoints are not available
    distance_to_waypoint = 0.0
    angle_to_waypoint = 0.0
    next_waypoint_x = 0.0
    next_waypoint_y = 0.0
    
    if current_waypoint:
        # Calculate distance to current waypoint
        distance_to_waypoint = math.sqrt(
            (location.x - current_waypoint.transform.location.x)**2 +
            (location.y - current_waypoint.transform.location.y)**2
        )
        
        # Calculate angle to current waypoint
        waypoint_direction = math.atan2(
            current_waypoint.transform.location.y - location.y,
            current_waypoint.transform.location.x - location.x
        )
        vehicle_direction = math.radians(rotation.yaw)
        angle_to_waypoint = math.degrees(waypoint_direction - vehicle_direction)
        
        # Normalize angle to [-180, 180]
        angle_to_waypoint = (angle_to_waypoint + 180) % 360 - 180
        
        # Get next waypoint location relative to vehicle
        if next_waypoint:
            # Transform next waypoint to vehicle's local coordinate system
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # Calculate relative position
            dx = next_waypoint.transform.location.x - vehicle_location.x
            dy = next_waypoint.transform.location.y - vehicle_location.y
            
            # Rotate to vehicle's coordinate system
            yaw_rad = math.radians(vehicle_rotation.yaw)
            next_waypoint_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            next_waypoint_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    
    # Combine all vehicle state information
    vehicle_state = np.array([
        location.x, location.y, location.z,
        rotation.pitch, rotation.yaw, rotation.roll,
        velocity.x, velocity.y, velocity.z,
        angular_velocity.x, angular_velocity.y, angular_velocity.z,
        speed,
        distance_to_waypoint,
        angle_to_waypoint,
        next_waypoint_x
    ], dtype=np.float32)
    
    return vehicle_state

def get_scenario_observation(vehicle, active_scenario):
    """Get scenario-specific observation
    
    Args:
        vehicle: CARLA vehicle object
        active_scenario: Active scenario object
        
    Returns:
        numpy.ndarray: Scenario-specific observation
    """
    # Default observation
    scenario_obs = np.zeros(15, dtype=np.float32)
    
    # If no active scenario, return default
    if active_scenario is None:
        return scenario_obs
    
    # Get scenario-specific observation
    scenario_obs = active_scenario.get_scenario_specific_obs()
    
    return scenario_obs