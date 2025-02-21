import carla
import numpy as np
from src.environment.carla_env import CarlaEnv

class ObstacleAvoidanceScenario:
    def __init__(self, env: CarlaEnv):
        self.env = env
        self.world = env.world
        self.obstacles = []
        self.obstacle_sensors = []
        
    def setup(self):
        """Setup the obstacle avoidance scenario"""
        # Spawn ego vehicle (handled by env)
        
        # Spawn static obstacles
        self._spawn_static_obstacles()
        
        # Spawn dynamic obstacles
        self._spawn_dynamic_obstacles()
        
        # Setup sensors
        self._setup_sensors()
    
    def _spawn_static_obstacles(self, num_obstacles=5):
        """Spawn static obstacles in the scene"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Use props as static obstacles
        props = [
            'static.prop.streetbarrier',
            'static.prop.box01',
            'static.prop.container'
        ]
        
        for _ in range(num_obstacles):
            # Randomly select obstacle blueprint
            bp_name = np.random.choice(props)
            bp = blueprint_library.find(bp_name)
            
            # Try to spawn the obstacle
            spawn_point = np.random.choice(spawn_points)
            # Adjust spawn point to be on the road
            spawn_point.location.z += 0.5
            
            obstacle = self.world.spawn_actor(bp, spawn_point)
            if obstacle is not None:
                self.obstacles.append(obstacle)
    
    def _spawn_dynamic_obstacles(self, num_obstacles=2):
        """Spawn dynamic obstacles that move in predictable patterns"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        for _ in range(num_obstacles):
            # Use vehicles as dynamic obstacles
            bp = blueprint_library.find('vehicle.tesla.model3')
            
            spawn_point = np.random.choice(spawn_points)
            vehicle = self.world.spawn_actor(bp, spawn_point)
            
            if vehicle is not None:
                self.obstacles.append(vehicle)
                
                # Set up simple waypoint navigation
                waypoints = self.world.get_map().generate_waypoints(2.0)
                waypoint = np.random.choice(waypoints)
                vehicle.set_transform(waypoint.transform)
                
                # Set up constant velocity movement
                vehicle.enable_constant_velocity(carla.Vector3D(5, 0, 0))
    
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
        if not self.obstacles:
            return np.zeros(5)
            
        # Get nearest obstacle and its state
        ego_location = self.env.vehicle.get_location()
        nearest_obstacle = min(self.obstacles,
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
        for obstacle in self.obstacles:
            if obstacle.is_alive:
                obstacle.destroy()
        self.obstacles.clear()
        
        for sensor in self.obstacle_sensors:
            if sensor.is_alive:
                sensor.destroy()
        self.obstacle_sensors.clear() 