import carla
import numpy as np
import random
import math


def generate_control_from_action(action):
    """Generate CARLA control from action
    
    Args:
        action: Action from the agent [steering, throttle/brake]
        
    Returns:
        carla.VehicleControl: CARLA vehicle control
    """
    # Extract steering and throttle/brake from action
    steering = float(action[0])
    throttle_brake = float(action[1])
    
    # Create control object
    control = carla.VehicleControl()
    
    # Set steering (clamp to [-1, 1])
    control.steer = max(-1.0, min(1.0, steering))
    
    # Set throttle and brake based on throttle_brake value
    if throttle_brake >= 0:
        control.throttle = max(0.0, min(1.0, throttle_brake))
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = max(0.0, min(1.0, -throttle_brake))
    
    # Set manual gear shift to False
    control.manual_gear_shift = False
    
    return control


def spawn_ego_vehicle(world, blueprint_name='vehicle.tesla.model3', spawn_point=None):
    """Spawn ego vehicle in the world
    
    Args:
        world: CARLA world object
        blueprint_name: Name of the vehicle blueprint
        spawn_point: Spawn point for the vehicle
        
    Returns:
        tuple: (vehicle, spawn_point)
    """
    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()
    
    # Get the vehicle blueprint
    vehicle_bp = blueprint_library.find(blueprint_name)
    
    # Set the vehicle color
    if vehicle_bp.has_attribute('color'):
        color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
        vehicle_bp.set_attribute('color', color)
    
    # Get a random spawn point if not provided
    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
    
    # Spawn the vehicle
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    
    # If spawning failed, try again with a different spawn point
    if vehicle is None:
        spawn_points = world.get_map().get_spawn_points()
        for i in range(10):  # Try 10 times
            spawn_point = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                break
    
    # If still failed, raise an error
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle after multiple attempts")
    
    return vehicle, spawn_point


def generate_random_waypoints(vehicle, world, num_waypoints=20, min_distance=5.0, max_distance=10.0):
    """Generate random waypoints for path following
    
    Args:
        vehicle: CARLA vehicle object
        world: CARLA world object
        num_waypoints: Number of waypoints to generate
        min_distance: Minimum distance between waypoints
        max_distance: Maximum distance between waypoints
        
    Returns:
        tuple: (waypoints, current_waypoint_idx)
    """
    # Get the map
    carla_map = world.get_map()
    
    # Get the vehicle's current location
    vehicle_location = vehicle.get_location()
    
    # Get the nearest waypoint to the vehicle
    current_waypoint = carla_map.get_waypoint(vehicle_location)
    
    # Generate a path of waypoints
    waypoints = [current_waypoint]
    
    # Generate additional waypoints
    for _ in range(num_waypoints):
        # Get next waypoints
        next_waypoints = waypoints[-1].next(random.uniform(min_distance, max_distance))
        
        # If there are no next waypoints, break
        if not next_waypoints:
            break
        
        # Choose a random next waypoint
        next_waypoint = random.choice(next_waypoints)
        
        # Add to the list
        waypoints.append(next_waypoint)
    
    return waypoints, 0


def check_decision_points(vehicle, world, threshold_distance=20.0):
    """Check if the vehicle is near a decision point (intersection, lane merge, etc.)
    
    Args:
        vehicle: CARLA vehicle object
        world: CARLA world object
        threshold_distance: Distance threshold for decision points
        
    Returns:
        bool: True if near a decision point, False otherwise
    """
    if vehicle is None:
        return False
        
    # Get the map
    carla_map = world.get_map()
    
    # Get the vehicle's current location
    vehicle_location = vehicle.get_location()
    
    # Get the nearest waypoint to the vehicle
    current_waypoint = carla_map.get_waypoint(vehicle_location)
    
    # Check if the waypoint is at a junction
    if current_waypoint.is_junction:
        return True
    
    # Check if there's a junction ahead within the threshold distance
    waypoint = current_waypoint
    distance = 0.0
    
    while distance < threshold_distance:
        # Get next waypoints
        next_waypoints = waypoint.next(2.0)  # 2.0 meters ahead
        
        # If there are no next waypoints, break
        if not next_waypoints:
            break
        
        # Choose the first next waypoint
        waypoint = next_waypoints[0]
        
        # Update distance
        distance += 2.0
        
        # Check if the waypoint is at a junction
        if waypoint.is_junction:
            return True
    
    # Check for lane merges or lane changes
    left_lane = current_waypoint.get_left_lane()
    right_lane = current_waypoint.get_right_lane()
    
    # If there's a lane to the left or right with the same road_id but different lane_id
    if (left_lane and left_lane.lane_id != current_waypoint.lane_id and 
        left_lane.road_id == current_waypoint.road_id):
        return True
    
    if (right_lane and right_lane.lane_id != current_waypoint.lane_id and 
        right_lane.road_id == current_waypoint.road_id):
        return True
    
    return False


def calculate_path_reward(vehicle, waypoints, current_waypoint_idx, waypoint_threshold=0.1, target_speed=30):
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
    
    # Normalize velocity vector
    velocity_direction = velocity_vector / target_speed
    
    # Get current and next waypoint
    current_waypoint = waypoints[current_waypoint_idx]
    next_waypoint = waypoints[current_waypoint_idx + 1]
    
    # Calculate path direction vector (from current to next waypoint)
    # Handle both CARLA Waypoint objects and simple waypoints with x,y attributes
    if hasattr(current_waypoint, 'transform'):
        # CARLA Waypoint objects
        current_loc = current_waypoint.transform.location
        next_loc = next_waypoint.transform.location
        path_vector = np.array([
            next_loc.x - current_loc.x,
            next_loc.y - current_loc.y
        ])
    else:
        # Simple waypoints with x,y attributes
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
        if hasattr(current_waypoint, 'transform'):
            # CARLA Waypoint
            waypoint_pos = np.array([current_waypoint.transform.location.x, current_waypoint.transform.location.y])
        else:
            # Simple waypoint
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
    path_reward = np.dot(velocity_direction, path_direction)


    # Add waypoint reaching bonus
    if waypoints and current_waypoint_idx < len(waypoints):
        ego_location = vehicle.get_location()
        
        # Get location of the current waypoint
        if hasattr(waypoints[current_waypoint_idx], 'transform'):
            # CARLA Waypoint
            waypoint_location = waypoints[current_waypoint_idx].transform.location
            distance = np.sqrt(
                (ego_location.x - waypoint_location.x) ** 2 +
                (ego_location.y - waypoint_location.y) ** 2
            )
        else:
            # Simple waypoint
            distance = np.sqrt(
                (ego_location.x - waypoints[current_waypoint_idx].x) ** 2 +
                (ego_location.y - waypoints[current_waypoint_idx].y) ** 2
            )
        
        # Additional reward for reaching waypoint
        if distance < waypoint_threshold:
            path_reward += 1.0
    
    return path_reward