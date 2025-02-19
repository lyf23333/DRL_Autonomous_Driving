


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


def main():
    # Connect to the CARLA simulator
    client = carla.Client("localhost", 2000)  # IP and port
    client.set_timeout(10.0)  # Set timeout for connection

        # Access the world object
    world = client.get_world()
    print("Connected to CARLA!")

    # Get the map name
    map_name = world.get_map().name
    print(f"Current map: {map_name}")



if __name__ == "__main__":
    main()