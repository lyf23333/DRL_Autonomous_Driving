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
