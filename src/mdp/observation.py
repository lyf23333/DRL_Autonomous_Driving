import numpy as np
import carla


def get_obs(vehicle, waypoints, current_waypoint_idx, waypoint_threshold, trust_interface, active_scenario):
    """Get current observation of the environment"""
    if vehicle is None:
        return {
            'vehicle_state': np.zeros(12),
            'recent_intervention': 0,
            'scenario_obs': np.zeros(20)
        }
    
    # Get vehicle state
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()
    control = vehicle.get_control()
    
    # Get path following info
    distance_to_waypoint = float('inf')
    angle_to_waypoint = 0.0
    next_waypoint_x = 0.0
    next_waypoint_y = 0.0
    
    if waypoints and current_waypoint_idx < len(waypoints):
        ego_transform = vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        
        # Get next waypoint
        next_waypoint = waypoints[current_waypoint_idx]
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
        if distance_to_waypoint < waypoint_threshold:
            current_waypoint_idx += 1
    
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
        trust_interface.get_intervention_observation()
        if trust_interface is not None else 0
    )
    
    # Get scenario-specific observations
    if active_scenario:
        scenario_obs = active_scenario.get_scenario_specific_obs()
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