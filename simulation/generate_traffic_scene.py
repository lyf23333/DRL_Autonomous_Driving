
import carla
import time
from utils import get_and_print_blueprints
import random


def main():
    # Connect to the CARLA simulator
    client = carla.Client("localhost", 2000)  # IP and port
    client.set_timeout(10.0)  # Set timeout for connection

    world = client.get_world()
    print("Connected to CARLA!")

    map_name = world.get_map().name
    print(f"Current map: {map_name}")

    # Load a lightweight map
    client.load_world("Town04")  # Smallest map, best for low GPU usage

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

    # Spawn camera and attach it to the vehicle
    # Camera will be mounted 1.5m forward and 2.0m up from vehicle center
    camera_bp = blueprint_library.find("sensor.camera.rgb")  # RGB Camera
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))  # Positioning
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Function to process images
    def process_image(image):
        image.save_to_disk("output/frame_%06d.png" % image.frame)

    camera.listen(lambda image: process_image(image))


    # Move forward for 5 seconds
    vehicle.apply_control(carla.VehicleControl(throttle=0.5))
    time.sleep(5)
    # Stop the vehicle
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

    # Make the pedestrian walk forward
    pedestrian.apply_control(carla.WalkerControl(direction=carla.Vector3D(1, 0, 0)))
    time.sleep(5)
    # Stop the pedestrian
    pedestrian.apply_control(carla.WalkerControl(direction=carla.Vector3D(0, 0, 0)))

    # Clean up
    vehicle.destroy()
    pedestrian.destroy()
    camera.destroy()

if __name__ == "__main__":
    main()