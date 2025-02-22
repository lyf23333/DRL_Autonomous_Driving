import carla
import numpy as np
from src.environment.carla_env import CarlaEnv

class LaneSwitchingScenario:
    def __init__(self, env: CarlaEnv):
        self.env = env
        self.world = env.world
        
    def setup(self):
        """Setup the lane switching scenario"""
        # Spawn ego vehicle (already handled by env)
        
        # Spawn other vehicles
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Spawn vehicle in adjacent lane
        vehicle_bp = blueprint_library.find('vehicle.toyota.prius')
        if len(spawn_points) > 1:
            spawn_point = spawn_points[1]
            # Adjust spawn point to be in adjacent lane
            spawn_point.location.x += 10  # Adjust based on lane width
            self.other_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Set autopilot for other vehicle
            self.other_vehicle.set_autopilot(True)
        else:
            raise ValueError("No spawn points found for other vehicle")
    
    def get_scenario_specific_obs(self):
        """Get scenario-specific observations"""
        if not hasattr(self, 'other_vehicle'):
            return np.zeros(5)
            
        other_vehicle_transform = self.other_vehicle.get_transform()
        other_vehicle_velocity = self.other_vehicle.get_velocity()
        
        return np.array([
            other_vehicle_transform.location.x,
            other_vehicle_transform.location.y,
            other_vehicle_transform.rotation.yaw,
            other_vehicle_velocity.x,
            other_vehicle_velocity.y
        ])
    
    def check_scenario_completion(self):
        """Check if the lane switching scenario is completed"""
        if not hasattr(self, 'env') or not hasattr(self, 'other_vehicle'):
            return False
            
        ego_vehicle = self.env.vehicle
        if ego_vehicle is None:
            return False
            
        # Get vehicle locations
        ego_location = ego_vehicle.get_transform().location
        other_location = self.other_vehicle.get_transform().location
        
        # Check if ego vehicle has successfully changed lanes
        # This is a simple check - you might want to make it more sophisticated
        lateral_distance = abs(ego_location.y - other_location.y)
        
        return lateral_distance > 5.0  # Assuming lane width is less than 5 meters
    
    def cleanup(self):
        """Clean up the scenario"""
        if hasattr(self, 'other_vehicle'):
            self.other_vehicle.destroy() 