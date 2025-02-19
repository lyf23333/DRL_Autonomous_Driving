
import carla

from utils import get_and_print_blueprints

def main():
    # Connect to the CARLA simulator
    client = carla.Client("localhost", 2000)  # IP and port
    client.set_timeout(10.0)  # Set timeout for connection

    world = client.get_world()
    print("Connected to CARLA!")

    map_name = world.get_map().name
    print(f"Current map: {map_name}")

    blueprint_library = get_and_print_blueprints(world)

    # Get the blueprint for a vehicle
    vehicle_blueprint = blueprint_library.find("vehicle.tesla.model3")
    print(f"Vehicle blueprint: {vehicle_blueprint.id}")

    # Get the blueprint for a pedestrian
    pedestrian_blueprint = blueprint_library.find("walker.pedestrian.0001")
    print(f"Pedestrian blueprint: {pedestrian_blueprint.id}")

    # Get spawn points for all actors
    spawn_points = world.get_map().get_spawn_points()
    
    # Spawn vehicle at first available spawn point
    vehicle_spawn_point = spawn_points[0]
    vehicle = world.spawn_actor(vehicle_blueprint, vehicle_spawn_point)
    print(f"Vehicle spawned at: {vehicle_spawn_point.location}")

    # Spawn pedestrian at second spawn point
    # Need to offset from road spawn point for pedestrian
    pedestrian_spawn_point = spawn_points[1]
    pedestrian_spawn_point.location.z += 1.0  # Lift slightly above ground
    pedestrian = world.spawn_actor(pedestrian_blueprint, pedestrian_spawn_point)
    print(f"Pedestrian spawned at: {pedestrian_spawn_point.location}")



if __name__ == "__main__":
    main()