import carla
import numpy as np
from src.environment.carla_env import CarlaEnv

class LaneSwitchingScenario:
    def __init__(self, env: CarlaEnv):
        self.env = env
        self.world = env.world
        self.other_vehicles = []
        self.target_speed = 10  # km/h for the slower vehicle
        self.initial_distance = 30.0  # meters - increased for safe spawning
        self.safe_distance = 10.0  # meters for safety checks
        self.lane_width = 3.5  # meters
        
    def setup(self):
        """Setup the lane switching scenario with a slower vehicle ahead"""
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return
            
        # Get ego vehicle's transform and waypoint
        ego_transform = self.env.vehicle.get_transform()
        ego_location = ego_transform.location
        ego_waypoint = self.world.get_map().get_waypoint(ego_location)
        
        # Get waypoint ahead of ego vehicle
        spawn_waypoint = ego_waypoint
        distance = 0
        
        # Find a suitable spawn point ahead
        while distance < self.initial_distance:
            next_waypoints = spawn_waypoint.next(5.0)  # Get waypoints 5m ahead
            if not next_waypoints:
                return
            spawn_waypoint = next_waypoints[0]
            distance = ego_location.distance(spawn_waypoint.transform.location)
        
        # Create spawn transform at the waypoint
        spawn_transform = spawn_waypoint.transform
        spawn_transform.location.z += 0.5  # Lift slightly to avoid collision
        
        # Spawn slower vehicle ahead
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.toyota.prius')
        
        # Set color for better visibility
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')  # Red color
            
        # Try to spawn the vehicle with collision checks
        slower_vehicle = None
        max_attempts = 5
        
        for _ in range(max_attempts):
            slower_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if slower_vehicle is not None:
                break
            # If spawn failed, try a bit further ahead
            next_waypoints = spawn_waypoint.next(5.0)
            if not next_waypoints:
                break
            spawn_waypoint = next_waypoints[0]
            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 0.5
        
        if slower_vehicle is not None:
            self.other_vehicles.append(slower_vehicle)
            
            # Set up the slower vehicle's behavior
            slower_vehicle.set_autopilot(True)
            
            # Get forward vector for velocity
            forward_vector = spawn_transform.get_forward_vector()
            
            # Set target velocity for slower vehicle
            target_speed_ms = self.target_speed / 3.6  # Convert km/h to m/s
            slower_vehicle.set_target_velocity(
                carla.Vector3D(
                    x=forward_vector.x * target_speed_ms,
                    y=forward_vector.y * target_speed_ms,
                    z=0
                )
            )
            
            # Add velocity controller to maintain slow speed
            self._setup_velocity_control(slower_vehicle)
            
            print(f"Successfully spawned slower vehicle at distance: {distance:.1f}m")
        else:
            print("Failed to spawn slower vehicle after multiple attempts")
    
    def _setup_velocity_control(self, vehicle):
        """Setup a velocity controller for the vehicle"""
        def velocity_control(weak_vehicle):
            vehicle = weak_vehicle()
            if vehicle is None:
                return
                
            # Get current velocity
            velocity = vehicle.get_velocity()
            speed_ms = np.sqrt(velocity.x**2 + velocity.y**2)
            target_speed_ms = self.target_speed / 3.6
            
            # Adjust throttle to maintain target speed
            if speed_ms < target_speed_ms:
                vehicle.apply_control(carla.VehicleControl(throttle=0.5))
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0))
        
        # Create weak reference to avoid circular reference
        import weakref
        weak_vehicle = weakref.ref(vehicle)
        
        # Add tick callback
        self.world.on_tick(lambda _: velocity_control(weak_vehicle))
    
    def get_scenario_specific_obs(self):
        """Get scenario-specific observations"""
        if not self.other_vehicles:
            return np.zeros(8)
            
        ego_vehicle = self.env.vehicle
        if ego_vehicle is None:
            return np.zeros(8)
            
        # Get ego vehicle state
        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = np.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        
        # Get lead vehicle state
        lead_vehicle = self.other_vehicles[0]
        lead_transform = lead_vehicle.get_transform()
        lead_velocity = lead_vehicle.get_velocity()
        lead_speed = np.sqrt(lead_velocity.x**2 + lead_velocity.y**2)
        
        # Calculate relative position and velocity
        relative_location = lead_transform.location - ego_transform.location
        distance = np.sqrt(relative_location.x**2 + relative_location.y**2)
        relative_speed = lead_speed - ego_speed
        
        # Get lane information
        ego_waypoint = self.world.get_map().get_waypoint(ego_transform.location)
        left_lane = ego_waypoint.get_left_lane()
        right_lane = ego_waypoint.get_right_lane()
        
        # Check if adjacent lanes exist
        left_lane_exists = left_lane is not None and left_lane.lane_type == carla.LaneType.Driving
        right_lane_exists = right_lane is not None and right_lane.lane_type == carla.LaneType.Driving
        
        return np.array([
            distance,              # Distance to lead vehicle
            relative_speed,        # Relative speed
            lead_speed,           # Lead vehicle speed
            ego_speed,            # Ego vehicle speed
            float(left_lane_exists),  # Left lane availability
            float(right_lane_exists), # Right lane availability
            relative_location.x,   # Relative x position
            relative_location.y    # Relative y position
        ])
    
    def check_scenario_completion(self):
        """Check if the lane switching scenario is completed"""
        if not self.other_vehicles or not hasattr(self.env, 'vehicle'):
            return False
            
        ego_vehicle = self.env.vehicle
        if ego_vehicle is None:
            return False
            
        # Get vehicle states
        ego_transform = ego_vehicle.get_transform()
        lead_transform = self.other_vehicles[0].get_transform()
        
        # Calculate relative position
        relative_location = lead_transform.location - ego_transform.location
        distance = np.sqrt(relative_location.x**2 + relative_location.y**2)
        
        # Get ego vehicle velocity
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = np.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        
        # Scenario is complete if:
        # 1. Successfully passed the slower vehicle (negative x distance)
        # 2. Maintained safe distance
        # 3. Achieved higher speed
        return (
            relative_location.x < 0 and  # Passed the vehicle
            distance > self.safe_distance and  # Safe distance
            ego_speed > (self.target_speed / 3.6)  # Higher speed
        )
    
    def cleanup(self):
        """Clean up the scenario"""
        for vehicle in self.other_vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.other_vehicles.clear() 