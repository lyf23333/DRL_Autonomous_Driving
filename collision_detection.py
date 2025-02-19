import carla
import time


def main():
    # Connect to CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get blueprints
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]
    collision_bp = blueprint_library.find("sensor.other.collision")

    # Spawn vehicle
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    # Attach collision sensor
    collision_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
    collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)

    # Collision event flag
    collision_detected = False

    # Define collision callback
    def collision_callback(event):
        global collision_detected
        collision_detected = True
        actor = event.other_actor  # Object that was hit
        impulse = event.normal_impulse  # Collision force

        print(f"Collision with: {actor.type_id}")
        print(f"Impact force: {impulse}")

    # Listen for collisions
    collision_sensor.listen(lambda event: collision_callback(event))

    # Move forward
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

    # Wait for collision
    while not collision_detected:
        time.sleep(0.1)

    # Stop the vehicle
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    print("Vehicle stopped due to collision!")

    # Cleanup
    collision_sensor.destroy()
    vehicle.destroy()
    print("Actors destroyed.")

if __name__ == "__main__":
    main()
