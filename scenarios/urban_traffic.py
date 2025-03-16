import carla
import numpy as np
from src.environment.carla_env import CarlaEnv

class UrbanTrafficScenario:
    """A scenario class that simulates urban traffic conditions in CARLA.
    
    This class creates a realistic urban environment with:
    - Multiple traffic vehicles driving autonomously
    - Pedestrians walking on sidewalks and crossing streets
    - Traffic lights at intersections
    
    The scenario spawns vehicles and pedestrians at valid spawn points while maintaining:
    - Minimum safe distances between actors
    - Realistic traffic patterns and behaviors
    - Proper traffic light synchronization
    
    The scenario tracks all spawned actors and provides cleanup functionality.
    """
    def __init__(self, env: CarlaEnv):
        self.env = env
        self.world = env.world
        self.vehicles = []
        self.walkers = []
        self._is_setup = False

    @property
    def is_setup(self):
        return self._is_setup
        
    def setup(self):
        """Setup the urban traffic scenario"""
        
        # Clean up any existing actors from previous runs
        self.cleanup()
        
        # Spawn other vehicles
        self._spawn_traffic_vehicles(num_vehicles=10)
        
        # Spawn pedestrians
        self._spawn_pedestrians(num_pedestrians=5)
        
        # Set up traffic lights
        self._setup_traffic_lights()

        self._is_setup = True
    
    def _spawn_traffic_vehicles(self, num_vehicles):
        """Spawn traffic vehicles in the scene"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Keep track of used spawn points
        used_spawn_points = set()
        min_distance = 15.0  # Minimum distance between vehicles
        
        for _ in range(min(num_vehicles, len(spawn_points))):
            # Find a suitable spawn point
            valid_spawn_point = None
            max_attempts = 20
            
            for _ in range(max_attempts):
                candidate_point = np.random.choice(spawn_points)
                
                # Check distance from other vehicles
                is_valid = True
                for used_point in used_spawn_points:
                    if candidate_point.location.distance(used_point.location) < min_distance:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_spawn_point = candidate_point
                    break
            
            if valid_spawn_point is None:
                continue
            
            # Randomly select vehicle blueprint
            bp = np.random.choice(blueprint_library.filter('vehicle.*.*'))
            
            # Try to spawn the vehicle
            vehicle = self.world.spawn_actor(bp, valid_spawn_point)
            
            if vehicle is not None:
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
                used_spawn_points.add(valid_spawn_point)
    
    def _spawn_pedestrians(self, num_pedestrians):
        """Spawn pedestrians in the scene"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Keep track of used spawn points
        used_spawn_points = set()
        min_distance = 10.0  # Minimum distance between pedestrians
        
        for _ in range(num_pedestrians):
            # Find a suitable spawn point
            valid_spawn_point = None
            max_attempts = 20
            
            for _ in range(max_attempts):
                candidate_point = np.random.choice(spawn_points)
                
                # Check distance from other pedestrians
                is_valid = True
                for used_point in used_spawn_points:
                    if candidate_point.location.distance(used_point.location) < min_distance:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_spawn_point = candidate_point
                    break
            
            if valid_spawn_point is None:
                continue
            
            # Randomly select pedestrian blueprint
            bp = np.random.choice(blueprint_library.filter('walker.pedestrian.*'))
            
            # Adjust spawn point for pedestrians
            valid_spawn_point.location.z += 1  # Raise spawn point to avoid collision
            walker = self.world.spawn_actor(bp, valid_spawn_point)
            
            if walker is not None:
                self.walkers.append(walker)
                used_spawn_points.add(valid_spawn_point)
                
                # Set up AI controller for the walker
                controller_bp = blueprint_library.find('controller.ai.walker')
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                controller.start()
                
                # Get a random location on the navigation mesh
                target_location = self.world.get_random_location_from_navigation()
                if target_location:
                    controller.go_to_location(target_location)
                    controller.set_max_speed(1.4)
    
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
        self.vehicles = []
        
        for walker in self.walkers:
            if walker.is_alive:
                walker.destroy()
        self.walkers.clear() 
        self.walkers = []