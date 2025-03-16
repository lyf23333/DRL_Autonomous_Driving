import numpy as np
import carla


def get_obs(vehicle, waypoints, current_waypoint_idx, waypoint_threshold, trust_interface, active_scenario):
    """Get current observation of the environment"""
    # Default observation with zeros
    default_vehicle_state = np.zeros(16)  # Increased size to accommodate 4 waypoints (8 values) + 12 original values
    
    if vehicle is None:
        return {
            'vehicle_state': default_vehicle_state,
            'recent_intervention': 0,
            'scenario_obs': np.zeros(20)
        }
    
    # Get vehicle state
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()
    control = vehicle.get_control()
    
    # Get vehicle transform
    ego_transform = vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()
    ego_right = ego_transform.get_right_vector()
    
    # Initialize waypoint information arrays
    relative_waypoints_xy = np.zeros((4, 2))  # Next 4 waypoints relative x positions
    distances_to_waypoints = np.zeros(4)  # Distances to next 4 waypoints
    angles_to_waypoints = np.zeros(4)  # Angles to next 4 waypoints
    
    # Process waypoints if available
    if waypoints and current_waypoint_idx < len(waypoints):
        # Process the next 4 waypoints (or as many as available)
        for i in range(4):
            waypoint_idx = current_waypoint_idx + i
            
            # Check if this waypoint exists
            if waypoint_idx < len(waypoints):
                next_waypoint = waypoints[waypoint_idx]
                
                # Calculate waypoint vector in world coordinates
                waypoint_vector_world = carla.Vector3D(
                    x=next_waypoint.x - ego_location.x,
                    y=next_waypoint.y - ego_location.y,
                    z=0.0
                )
                
                # Transform to vehicle's local coordinate system
                # Forward = x-axis, Right = y-axis
                relative_x = (waypoint_vector_world.x * ego_forward.x + 
                             waypoint_vector_world.y * ego_forward.y)
                relative_y = (waypoint_vector_world.x * ego_right.x + 
                             waypoint_vector_world.y * ego_right.y)
                
                # Store relative coordinates
                relative_waypoints_xy[i] = np.array([relative_x, relative_y])
                
                # Calculate distance to waypoint
                distances_to_waypoints[i] = np.sqrt(relative_x**2 + relative_y**2)
                
                # Calculate angle to waypoint (in vehicle's frame)
                angles_to_waypoints[i] = np.arctan2(relative_y, relative_x)
                
        # Update waypoint index if close enough to the current waypoint
        if distances_to_waypoints[0] < waypoint_threshold:
            current_waypoint_idx += 1
    
    # Combine all vehicle state information
    vehicle_state = np.array([
        velocity.x, velocity.y,              # Linear velocity
        acceleration.x, acceleration.y,       # Linear acceleration
        angular_velocity.z,                   # Angular velocity (yaw rate)
        control.steer,                       # Current steering
        control.throttle,                    # Current throttle
        control.brake,                       # Current brake
        # Next 4 waypoints (relative positions, distances, and angles)
        *relative_waypoints_xy.flatten()
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