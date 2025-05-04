import carla
import numpy as np
from src.environment.carla_env import CarlaEnv

class EmergencyBrakingScenario:
    """A scenario that tests emergency braking capabilities.
    
    This scenario creates situations requiring emergency braking by:
    - Suddenly spawning vehicles that cut in front
    - Creating unexpected obstacles in the path
    - Simulating emergency stop scenarios
    """
    
    def __init__(self, env: CarlaEnv):
        self.env = env
        self.world = env.world
        self.hazard_vehicle = None
        self.spawn_distance = 30.0  # meters
        self.critical_distance = 20.0  # distance at which hazard appears
        self.safe_distance = 5.0  # minimum safe distance
        self._tick_callbacks = []
        self.hazard_triggered = False
        self.start_time = None
        self.reaction_time = None
        self._is_setup = False

    @property
    def is_setup(self):
        return self._is_setup
        
    def setup(self):
        """Setup the emergency braking scenario"""
        # Clean up any existing actors
        self.cleanup()
        
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return
            
        # Get ego vehicle's transform
        ego_transform = self.env.vehicle.get_transform()
        ego_location = ego_transform.location
        
        # Calculate spawn point in front of ego vehicle
        forward_vector = ego_transform.get_forward_vector()
        spawn_location = carla.Location(
            x=ego_location.x + forward_vector.x * self.spawn_distance,
            y=ego_location.y + forward_vector.y * self.spawn_distance,
            z=ego_location.z + 0.5
        )
        
        # Get the waypoint on the road
        spawn_waypoint = self.world.get_map().get_waypoint(spawn_location)
        if spawn_waypoint is None:
            print("Failed to find valid waypoint for hazard vehicle")
            return
            
        # Create spawn transform
        spawn_transform = spawn_waypoint.transform
        spawn_transform.location.z += 0.5  # Lift slightly to avoid collision
        
        # Spawn hazard vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')  # Red color
            
        self.hazard_vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
        
        if self.hazard_vehicle is not None:
            # Set initial velocity (faster than ego vehicle)
            target_speed_ms = 7.5 / 3.6  # 30 km/h to m/s
            road_direction = spawn_transform.get_forward_vector()
            self.hazard_vehicle.set_target_velocity(carla.Vector3D(
                x=road_direction.x * target_speed_ms,
                y=road_direction.y * target_speed_ms,
                z=0
            ))
            
            # Setup hazard behavior
            self._setup_hazard_behavior()
            print("Successfully spawned hazard vehicle")
        else:
            print("Failed to spawn hazard vehicle")

        self._is_setup = True
    
    def _setup_hazard_behavior(self):
        """Setup the hazard vehicle's behavior"""
        def hazard_control(weak_vehicle):
            vehicle = weak_vehicle()
            if vehicle is None or not vehicle.is_alive:
                return
                
            if not self.hazard_triggered:
                # Check distance to ego vehicle
                ego_location = self.env.vehicle.get_location()
                hazard_location = vehicle.get_location()
                distance = ego_location.distance(hazard_location)
                
                if distance <= self.critical_distance:
                    # Trigger emergency braking scenario
                    self.hazard_triggered = True
                    self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds
                    
                    # Sudden brake
                    control = carla.VehicleControl(throttle=0.0, brake=1.0)
                    vehicle.apply_control(control)
        
        # Create weak reference
        import weakref
        weak_vehicle = weakref.ref(self.hazard_vehicle)
        
        # Add tick callback
        callback_id = self.world.on_tick(lambda _: hazard_control(weak_vehicle))
        self._tick_callbacks.append(callback_id)
    
    def get_scenario_specific_obs(self):
        """Get scenario-specific observations"""
        if self.hazard_vehicle is None or not self.hazard_vehicle.is_alive:
            return np.zeros(6)
            
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return np.zeros(6)
            
        # Get ego vehicle state
        ego_transform = self.env.vehicle.get_transform()
        ego_velocity = self.env.vehicle.get_velocity()
        ego_speed = np.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        
        # Get hazard vehicle state
        hazard_transform = self.hazard_vehicle.get_transform()
        hazard_velocity = self.hazard_vehicle.get_velocity()
        hazard_speed = np.sqrt(hazard_velocity.x**2 + hazard_velocity.y**2)
        
        # Calculate relative position
        relative_location = hazard_transform.location - ego_transform.location
        distance = np.sqrt(relative_location.x**2 + relative_location.y**2)
        
        return np.array([
            distance,              # Distance to hazard vehicle
            ego_speed,            # Ego vehicle speed
            hazard_speed,         # Hazard vehicle speed
            float(self.hazard_triggered),  # Whether hazard has been triggered
            relative_location.x,   # Relative x position
            relative_location.y    # Relative y position
        ])
    
    def check_scenario_completion(self):
        """Check if the emergency braking scenario is completed"""
        if not self.hazard_triggered:
            return False
            
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return False
            
        if self.hazard_vehicle is None or not self.hazard_vehicle.is_alive:
            return False
            
        # Get current states
        ego_velocity = self.env.vehicle.get_velocity()
        ego_speed = np.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        
        ego_location = self.env.vehicle.get_location()
        hazard_location = self.hazard_vehicle.get_location()
        distance = ego_location.distance(hazard_location)
        
        # Calculate reaction time if not already set
        if self.reaction_time is None and ego_speed < 1.0:  # Speed less than 1 m/s
            current_time = self.world.get_snapshot().timestamp.elapsed_seconds
            self.reaction_time = current_time - self.start_time
        
        # Scenario is complete if:
        # 1. Vehicle has come to almost complete stop
        # 2. Maintained safe distance from hazard
        # 3. No collision occurred
        return (
            ego_speed < 1.0 and  # Almost stopped
            distance >= self.safe_distance  # Maintained safe distance
        )
    
    def cleanup(self):
        """Clean up the scenario"""
        # Remove tick callbacks
        if hasattr(self, '_tick_callbacks'):
            for callback_id in self._tick_callbacks:
                try:
                    self.world.remove_on_tick(callback_id)
                except:
                    pass
            self._tick_callbacks = []
        
        # Destroy hazard vehicle
        if self.hazard_vehicle is not None and self.hazard_vehicle.is_alive:
            try:
                self.hazard_vehicle.destroy()
            except:
                print("Warning: Failed to destroy hazard vehicle")
        
        self.hazard_vehicle = None
        self.hazard_triggered = False
        self.start_time = None
        self.reaction_time = None
        
        # Wait for cleanup
        self.world.tick() 