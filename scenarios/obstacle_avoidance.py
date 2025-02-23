import carla
import numpy as np
from src.environment.carla_env import CarlaEnv

class ObstacleAvoidanceScenario:
    """A scenario class that sets up an obstacle avoidance course in CARLA.
    
    This class creates a challenging environment with both static and dynamic obstacles:
    - Static obstacles: Street barriers, boxes, and containers placed at random valid locations
    - Dynamic vehicles: Other vehicles driving autonomously on the road
    - Dynamic walkers: Pedestrians walking randomly through the environment
    
    The scenario is configured via a dictionary that specifies:
    - Number of each type of obstacle
    - Minimum spawn distances from ego vehicle
    - Available models/props for each obstacle type
    
    The scenario tracks all spawned actors and provides cleanup functionality.
    """
    def __init__(self, env: CarlaEnv):
        self.env = env
        self.world = env.world
        
        # Separate lists for different types of obstacles
        self.static_obstacles = []
        self.dynamic_vehicles = []
        self.dynamic_walkers = []
        self.walker_controllers = []  # Keep track of walker controllers
        self.obstacle_sensors = []
        
        # Spawn configuration
        self.config = {
            'static_obstacles': {
                'num': 5,
                'min_distance': 10.0,
                'props': [
                    'static.prop.streetbarrier',
                    'static.prop.box01',
                    'static.prop.container'
                ]
            },
            'dynamic_vehicles': {
                'num': 2,
                'min_distance': 20.0,
                'models': [
                    'vehicle.tesla.model3',
                    'vehicle.audi.a2'
                ]
            },
            'dynamic_walkers': {
                'num': 2,
                'min_distance': 15.0,
                'models': [
                    'walker.pedestrian.0013',
                    'walker.pedestrian.0021'
                ]
            }
        }
        
        # Keep track of all used spawn points
        self.used_spawn_points = set()
    
    def setup(self):
        """Setup the obstacle avoidance scenario"""
        # Clear any existing obstacles
        self.cleanup()
        
        # Spawn static obstacles
        self._spawn_static_obstacles()
        
        # Spawn dynamic obstacles
        self._spawn_dynamic_vehicles()
        self._spawn_dynamic_walkers()
        
        # Setup sensors
        self._setup_sensors()
    
    def _find_valid_spawn_point(self, min_distance):
        """Find a valid spawn point that maintains minimum distance from all obstacles"""
        spawn_points = self.world.get_map().get_spawn_points()
        max_attempts = 20
        
        for _ in range(max_attempts):
            candidate_point = np.random.choice(spawn_points)
            
            # Check distance from all existing obstacles
            is_valid = True
            for used_point in self.used_spawn_points:
                if candidate_point.location.distance(used_point.location) < min_distance:
                    is_valid = False
                    break
            
            if is_valid:
                return candidate_point
        
        return None
    
    def _spawn_static_obstacles(self):
        """Spawn static obstacles in the scene"""
        blueprint_library = self.world.get_blueprint_library()
        config = self.config['static_obstacles']
        
        for _ in range(config['num']):
            # Randomly select obstacle blueprint
            bp_name = np.random.choice(config['props'])
            bp = blueprint_library.find(bp_name)
            
            # Find a suitable spawn point
            valid_spawn_point = self._find_valid_spawn_point(config['min_distance'])
            
            if valid_spawn_point is None:
                continue
            
            # Adjust spawn point
            valid_spawn_point.location.z += 0.5
            valid_spawn_point.rotation.yaw += np.random.uniform(-20, 20)
            
            obstacle = self.world.spawn_actor(bp, valid_spawn_point)
            if obstacle is not None:
                self.static_obstacles.append(obstacle)
                self.used_spawn_points.add(valid_spawn_point)
    
    def _spawn_dynamic_vehicles(self):
        """Spawn dynamic vehicle obstacles"""
        blueprint_library = self.world.get_blueprint_library()
        config = self.config['dynamic_vehicles']
        
        for _ in range(config['num']):
            # Randomly select vehicle blueprint
            bp_name = np.random.choice(config['models'])
            bp = blueprint_library.find(bp_name)
            
            # Find a suitable spawn point
            valid_spawn_point = self._find_valid_spawn_point(config['min_distance'])
            
            if valid_spawn_point is None:
                continue
            
            vehicle = self.world.spawn_actor(bp, valid_spawn_point)
            if vehicle is not None:
                self.dynamic_vehicles.append(vehicle)
                self.used_spawn_points.add(valid_spawn_point)
                vehicle.set_autopilot(True)
    
    def _spawn_dynamic_walkers(self):
        """Spawn dynamic walker obstacles"""
        blueprint_library = self.world.get_blueprint_library()
        config = self.config['dynamic_walkers']
        
        for _ in range(config['num']):
            # Randomly select walker blueprint
            bp_name = np.random.choice(config['models'])
            walker_bp = blueprint_library.find(bp_name)
            
            # Find a suitable spawn point
            valid_spawn_point = self._find_valid_spawn_point(config['min_distance'])
            
            if valid_spawn_point is None:
                continue
            
            # Adjust spawn point for pedestrians
            valid_spawn_point.location.z += 1
            walker = self.world.spawn_actor(walker_bp, valid_spawn_point)
            
            if walker is not None:
                self.dynamic_walkers.append(walker)
                self.used_spawn_points.add(valid_spawn_point)
                
                # Create and setup walker controller
                controller_bp = blueprint_library.find('controller.ai.walker')
                controller = self.world.spawn_actor(
                    controller_bp, carla.Transform(), attach_to=walker)
                
                if controller is not None:
                    self.walker_controllers.append(controller)
                    controller.start()
                    target_location = self.world.get_random_location_from_navigation()
                    if target_location:
                        controller.go_to_location(target_location)
                        controller.set_max_speed(1.4)

    def _setup_sensors(self):
        """Setup sensors for obstacle detection"""
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return
            
        blueprint_library = self.world.get_blueprint_library()
        
        # Setup LIDAR sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '90000')
        lidar_bp.set_attribute('rotation_frequency', '40')
        lidar_bp.set_attribute('range', '20')
        
        # Attach LIDAR to vehicle
        lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar = self.world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=self.env.vehicle
        )
        self.obstacle_sensors.append(lidar)
        
        # Add collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.env.vehicle
        )
        self.obstacle_sensors.append(collision_sensor)
    
    def get_scenario_specific_obs(self):
        """Get scenario-specific observations"""
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return np.zeros(5)
            
        ego_location = self.env.vehicle.get_location()
        
        # Combine all obstacles
        all_obstacles = (
            self.static_obstacles +
            self.dynamic_vehicles +
            self.dynamic_walkers
        )
        
        if not all_obstacles:
            return np.zeros(5)
        
        # Get nearest obstacle and its state
        nearest_obstacle = min(all_obstacles,
                             key=lambda o: ego_location.distance(o.get_location()))
        
        # Get obstacle state
        transform = nearest_obstacle.get_transform()
        
        # For static obstacles, velocity will be zero
        velocity = getattr(nearest_obstacle, 'get_velocity', lambda: carla.Vector3D(0,0,0))()
        
        return np.array([
            transform.location.x,
            transform.location.y,
            transform.rotation.yaw,
            velocity.x,
            velocity.y
        ])
    
    def check_scenario_completion(self):
        """Check if the obstacle avoidance scenario is completed"""
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return False
            
        # Example completion criteria:
        # - Reached destination
        # - Avoided all obstacles
        # - No collisions occurred
        return False  # Implement actual completion criteria
    
    def cleanup(self):
        """Clean up the scenario"""
        # Clean up static obstacles
        for obstacle in self.static_obstacles:
            if obstacle.is_alive:
                obstacle.destroy()
        self.static_obstacles.clear()
        
        # Clean up dynamic vehicles
        for vehicle in self.dynamic_vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.dynamic_vehicles.clear()
        
        # Clean up walker controllers first
        for controller in self.walker_controllers:
            if controller.is_alive:
                controller.stop()
                controller.destroy()
        self.walker_controllers.clear()
        
        # Clean up dynamic walkers
        for walker in self.dynamic_walkers:
            if walker.is_alive:
                walker.destroy()
        self.dynamic_walkers.clear()
        
        # Clean up sensors
        for sensor in self.obstacle_sensors:
            if sensor.is_alive:
                sensor.destroy()
        self.obstacle_sensors.clear()
        
        # Clear used spawn points
        self.used_spawn_points.clear()

    def _spawn_dynamic_obstacles(self, num_obstacles=2):
        """Spawn dynamic obstacles that move in predictable patterns"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Keep track of used spawn points
        used_spawn_points = set()
        min_distance = 20.0  # Larger minimum distance for dynamic obstacles

        vehicle_props = [
            "vehicle.tesla.model3",
            "vehicle.audi.a2"
        ]
        
        pedestrian_props = [
            "walker.pedestrian.0013",
            "walker.pedestrian.0021"
        ]
        
        for _ in range(num_obstacles):
            # Find a suitable spawn point
            valid_spawn_point = None
            max_attempts = 20
            
            for _ in range(max_attempts):
                candidate_point = np.random.choice(spawn_points)
                
                # Check distance from other obstacles
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
                
            # Randomly choose between vehicle and pedestrian
            if np.random.rand() > 0.5:
                # Use vehicles as dynamic obstacles
                bp_name = np.random.choice(vehicle_props)
                bp = blueprint_library.find(bp_name)
                
                vehicle = self.world.spawn_actor(bp, valid_spawn_point)
                
                if vehicle is not None:
                    self.obstacles.append(vehicle)
                    used_spawn_points.add(valid_spawn_point)
                    vehicle.set_autopilot(True)  # Enable autopilot
            else:
                # Use pedestrians as dynamic obstacles
                bp_name = np.random.choice(pedestrian_props)
                walker_bp = blueprint_library.find(bp_name)
                
                # Adjust spawn point for pedestrians
                valid_spawn_point.location.z += 1
                walker = self.world.spawn_actor(walker_bp, valid_spawn_point)
                
                if walker is not None:
                    self.obstacles.append(walker)
                    used_spawn_points.add(valid_spawn_point)
                    
                    # Create a walker controller
                    walker_controller_bp = blueprint_library.find('controller.ai.walker')
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), attach_to=walker)
                    
                    # Start the walker
                    walker_controller.start()
                    walker_controller.go_to_location(
                        self.world.get_random_location_from_navigation())
                    walker_controller.set_max_speed(1.4)  # Set walking speed 