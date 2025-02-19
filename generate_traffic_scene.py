


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Connect to the CARLA simulator
client = carla.Client("localhost", 2000)  # IP and port
client.set_timeout(10.0)  # Set timeout for connection

# Access the world object
world = client.get_world()
print("Connected to CARLA!")

# Get the map name
map_name = world.get_map().name
print(f"Current map: {map_name}")

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# List all available blueprints
print("\nAvailable blueprints:")
for blueprint in blueprint_library:
    print(f"- {blueprint.id}")

# You can also filter blueprints by type
vehicle_blueprints = blueprint_library.filter("vehicle.*")
walker_blueprints = blueprint_library.filter("walker.pedestrian.*")

print("\nVehicle blueprints:")
for bp in vehicle_blueprints:
    print(f"- {bp.id}")

print("\nPedestrian blueprints:") 
for bp in walker_blueprints:
    print(f"- {bp.id}")
