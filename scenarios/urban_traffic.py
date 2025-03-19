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
    
    def get_observation(self):
        """Get scenario-specific observations for the three nearest vehicles
        
        Returns:
            numpy.ndarray: Array of shape (15,) containing information about the three nearest vehicles:
                - For each vehicle (3 vehicles):
                    - Relative position x (in ego vehicle's coordinate frame)
                    - Relative position y (in ego vehicle's coordinate frame)
                    - Relative velocity x (in ego vehicle's coordinate frame)
                    - Relative velocity y (in ego vehicle's coordinate frame)
                    - Distance to the vehicle
                
        Note: Vehicles more than 20 meters away will have all their observation values set to 0.
        """
        # Initialize empty observation with zeros
        num_nearest_vehicles = 3
        obs = np.zeros(num_nearest_vehicles * 5)  # 5 values for each of 3 vehicles
        
        # If no vehicles or ego vehicle doesn't exist, return zeros
        if not self.vehicles or not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return obs
            
        # Get ego vehicle information
        ego_vehicle = self.env.vehicle
        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()
        ego_right = ego_transform.get_right_vector()
        
        # Calculate distances to all vehicles
        vehicle_distances = []
        for i, vehicle in enumerate(self.vehicles):
            # Skip if it's the ego vehicle or if vehicle is not alive
            if vehicle.id == ego_vehicle.id or not vehicle.is_alive:
                continue
                
            # Calculate distance
            distance = ego_location.distance(vehicle.get_location())
            
            # Only consider vehicles within 20 meters
            if distance <= 20.0:
                vehicle_distances.append((distance, i))
        
        # Sort by distance and take the three nearest
        vehicle_distances.sort()
        nearest_indices = [idx for _, idx in vehicle_distances[:3]]
        
        # Process the three nearest vehicles (or fewer if there aren't three)
        for i, idx in enumerate(nearest_indices):
            if i >= num_nearest_vehicles:  # Only process up to num_nearest_vehicles vehicles
                break
                
            vehicle = self.vehicles[idx]
            
            # Get vehicle state
            vehicle_location = vehicle.get_location()
            vehicle_velocity = vehicle.get_velocity()
            
            # Calculate relative position vector in world coordinates
            rel_location = carla.Vector3D(
                x=vehicle_location.x - ego_location.x,
                y=vehicle_location.y - ego_location.y,
                z=0.0
            )
            
            # Transform to ego vehicle's local coordinate system
            # Forward = x-axis, Right = y-axis
            rel_x = (rel_location.x * ego_forward.x + rel_location.y * ego_forward.y)
            rel_y = (rel_location.x * ego_right.x + rel_location.y * ego_right.y)
            
            # Calculate relative velocity
            rel_vel_x = vehicle_velocity.x - ego_vehicle.get_velocity().x
            rel_vel_y = vehicle_velocity.y - ego_vehicle.get_velocity().y
            
            # Transform relative velocity to ego vehicle's coordinate system
            rel_vel_local_x = (rel_vel_x * ego_forward.x + rel_vel_y * ego_forward.y)
            rel_vel_local_y = (rel_vel_x * ego_right.x + rel_vel_y * ego_right.y)
            
            # Calculate distance
            distance = ego_location.distance(vehicle_location)
            
            # Store in observation array (5 values per vehicle)
            start_idx = i * 5
            obs[start_idx:start_idx+5] = [
                rel_x,              # Relative position x
                rel_y,              # Relative position y
                rel_vel_local_x,    # Relative velocity x
                rel_vel_local_y,    # Relative velocity y
                distance            # Distance to vehicle
            ]
        
        return obs
    
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