import carla
import numpy as np
from src.environment.carla_env import CarlaEnv

class UrbanTrafficScenario:
    def __init__(self, env: CarlaEnv):
        self.env = env
        self.world = env.world
        self.vehicles = []
        self.walkers = []
        
    def setup(self):
        """Setup the urban traffic scenario"""
        # Spawn ego vehicle (handled by env)
        
        # Spawn other vehicles
        self._spawn_traffic_vehicles(num_vehicles=10)
        
        # Spawn pedestrians
        self._spawn_pedestrians(num_pedestrians=5)
        
        # Set up traffic lights
        self._setup_traffic_lights()
    
    def _spawn_traffic_vehicles(self, num_vehicles):
        """Spawn traffic vehicles in the scene"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        for _ in range(min(num_vehicles, len(spawn_points))):
            # Randomly select vehicle blueprint
            bp = np.random.choice(blueprint_library.filter('vehicle.*.*'))
            
            # Try to spawn the vehicle
            spawn_point = np.random.choice(spawn_points)
            vehicle = self.world.spawn_actor(bp, spawn_point)
            
            if vehicle is not None:
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
    
    def _spawn_pedestrians(self, num_pedestrians):
        """Spawn pedestrians in the scene"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        for _ in range(num_pedestrians):
            # Randomly select pedestrian blueprint
            bp = np.random.choice(blueprint_library.filter('walker.pedestrian.*'))
            
            # Try to spawn the pedestrian
            spawn_point = np.random.choice(spawn_points)
            spawn_point.location.z += 1  # Raise spawn point to avoid collision
            walker = self.world.spawn_actor(bp, spawn_point)
            
            if walker is not None:
                self.walkers.append(walker)
                
                # Set up AI controller for the walker
                controller_bp = blueprint_library.find('controller.ai.walker')
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                controller.start()
                controller.go_to_location(self.world.get_random_location_from_navigation())
    
    def _setup_traffic_lights(self):
        """Configure traffic lights in the scene"""
        # Get all traffic lights
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        
        # Set up traffic light timing
        for traffic_light in traffic_lights:
            traffic_light.set_green_time(5.0)
            traffic_light.set_yellow_time(2.0)
            traffic_light.set_red_time(5.0)
    
    def get_scenario_specific_obs(self):
        """Get scenario-specific observations"""
        # Get nearest vehicle and its state
        if not self.vehicles:
            return np.zeros(5)
            
        ego_location = self.env.vehicle.get_location()
        nearest_vehicle = min(self.vehicles, 
                            key=lambda v: ego_location.distance(v.get_location()))
        
        # Get nearest vehicle state
        transform = nearest_vehicle.get_transform()
        velocity = nearest_vehicle.get_velocity()
        
        return np.array([
            transform.location.x,
            transform.location.y,
            transform.rotation.yaw,
            velocity.x,
            velocity.y
        ])
    
    def check_scenario_completion(self):
        """Check if the urban traffic scenario is completed"""
        # Example completion criteria:
        # - Reached destination
        # - Maintained safe distance from other vehicles
        # - Obeyed traffic rules
        return False  # Implement actual completion criteria
    
    def cleanup(self):
        """Clean up the scenario"""
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.vehicles.clear()
        
        for walker in self.walkers:
            if walker.is_alive:
                walker.destroy()
        self.walkers.clear() 