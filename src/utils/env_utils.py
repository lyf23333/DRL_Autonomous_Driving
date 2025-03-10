import carla
import numpy as np


def generate_control_from_action(action):
    """
    Generate a control object from an action
    """
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

    return control


def spawn_ego_vehicle(world):
    """
    Spawn an ego vehicle at a given spawn point
    """
    # Spawn vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    
    # Find a valid spawn point
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise ValueError("No spawn points available in the map")

    # Try a few random spawn points
    for _ in range(10):
        spawn_point = np.random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            break
    
    # If still failed, try all spawn points sequentially
    if vehicle is None:
        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                break
    
    # If all spawn points failed, raise an error
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle at any spawn point")

    return vehicle, spawn_point


def generate_random_waypoints(vehicle, world, path_length=10):
    """Generate random waypoints for path following"""
    if vehicle is None:
        return
        
    # Clear existing waypoints
    waypoints = []
    current_waypoint_idx = 0
    
    # Get the current waypoint on the road
    current_waypoint = world.get_map().get_waypoint(vehicle.get_location())
    if current_waypoint is None:
        return
        
    # Generate a path by following the road
    next_waypoint = current_waypoint
    for _ in range(path_length):
        # Get next waypoint along the road
        next_waypoints = next_waypoint.next(5.0)  # 5 meters between waypoints
        if not next_waypoints:
            break
            
        # At intersections, randomly choose a direction
        if len(next_waypoints) > 1:
            next_waypoint = np.random.choice(next_waypoints)
        else:
            next_waypoint = next_waypoints[0]
            
        # Convert CARLA waypoint to a simple object with x, y attributes
        # This is needed because the observation space expects simple coordinates
        class SimpleWaypoint:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                
        simple_waypoint = SimpleWaypoint(
            next_waypoint.transform.location.x,
            next_waypoint.transform.location.y
        )
        
        waypoints.append(simple_waypoint)
        
    print(f"Generated {len(waypoints)} waypoints for path following")

    return waypoints, current_waypoint_idx


def process_collision(event, env):
    """
    Process collision events from CARLA
    
    Args:
        event: The collision event from CARLA
        env: The environment instance to update
        
    Returns:
        collision_detected: Whether a collision was detected
        collision_impulse: The impulse vector of the collision
    """
    # Set collision flag
    collision_detected = True
    
    # Get collision details
    collision_actor = event.other_actor
    impulse = event.normal_impulse
    intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    
    # Create collision impulse array
    collision_impulse = np.array([impulse.x, impulse.y, impulse.z])
    
    # Log collision details
    actor_type = collision_actor.type_id if hasattr(collision_actor, 'type_id') else "unknown"
    print(f"Collision detected with {actor_type}, intensity: {intensity:.2f}")
    
    return collision_detected, collision_impulse


def calculate_path_reward(vehicle, waypoints, current_waypoint_idx):
    """
    Calculate path reward based on alignment of vehicle velocity with path direction.
    
    The reward is higher when the vehicle's velocity vector is aligned with the
    direction from the current waypoint to the next waypoint.
    
    Args:
        vehicle: CARLA vehicle actor
        waypoints: List of waypoints
        current_waypoint_idx: Index of the current waypoint
        
    Returns:
        path_reward: Reward value between -1.0 and 1.0
    """
    if vehicle is None or not waypoints or current_waypoint_idx >= len(waypoints) - 1:
        return 0.0
    
    # Get vehicle velocity
    velocity = vehicle.get_velocity()
    velocity_vector = np.array([velocity.x, velocity.y])
    velocity_magnitude = np.linalg.norm(velocity_vector)
    
    # If vehicle is not moving, no direction reward
    if velocity_magnitude < 0.1:  # Threshold to consider vehicle as stationary
        return 0.0
    
    # Normalize velocity vector
    velocity_direction = velocity_vector / velocity_magnitude
    
    # Get current and next waypoint
    current_waypoint = waypoints[current_waypoint_idx]
    next_waypoint = waypoints[current_waypoint_idx + 1]
    
    # Calculate path direction vector (from current to next waypoint)
    path_vector = np.array([
        next_waypoint.x - current_waypoint.x,
        next_waypoint.y - current_waypoint.y
    ])
    path_magnitude = np.linalg.norm(path_vector)
    
    # If waypoints are too close, use the direction to the current waypoint
    if path_magnitude < 0.1:
        # Get vehicle position
        vehicle_location = vehicle.get_location()
        vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])
        
        # Calculate direction to current waypoint
        waypoint_pos = np.array([current_waypoint.x, current_waypoint.y])
        path_vector = waypoint_pos - vehicle_pos
        path_magnitude = np.linalg.norm(path_vector)
        
        # If still too close, return neutral reward
        if path_magnitude < 0.1:
            return 0.0
    
    # Normalize path direction vector
    path_direction = path_vector / path_magnitude
    
    # Calculate dot product between velocity direction and path direction
    # This gives the cosine of the angle between the two vectors
    # 1.0 = perfectly aligned, 0.0 = perpendicular, -1.0 = opposite direction
    alignment = np.dot(velocity_direction, path_direction)
    
    # Scale the reward to emphasize alignment
    # This gives a higher reward for alignment and a penalty for going in the wrong direction
    path_reward = alignment
    
    return path_reward
