import carla

def get_and_print_blueprints(world: carla.World) -> carla.BlueprintLibrary:
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

    return blueprint_library