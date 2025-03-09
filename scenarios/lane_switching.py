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
            
        # Clean up any existing actors from previous runs
        self.cleanup()
        
        # Get ego vehicle's transform and waypoint
        ego_transform = self.env.vehicle.get_transform()
        ego_location = ego_transform.location
        ego_waypoint = self.world.get_map().get_waypoint(ego_location)
        
        # Calculate spawn point directly in front of ego vehicle
        forward_vector = ego_transform.get_forward_vector()
        
        # Create spawn transform
        spawn_location = carla.Location(
            x=ego_location.x + forward_vector.x * self.initial_distance,
            y=ego_location.y + forward_vector.y * self.initial_distance,
            z=ego_location.z + 0.5
        )
        
        # Get the waypoint on the road at this location
        spawn_waypoint = self.world.get_map().get_waypoint(spawn_location)
        if spawn_waypoint is None:
            print("Failed to find valid waypoint for spawn location")
            return
            
        # Create spawn transform using waypoint (to ensure vehicle is on the road)
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
        spawn_offset = 0
        
        for attempt in range(max_attempts):
            # Adjust spawn point slightly forward on each attempt
            current_spawn_transform = carla.Transform(
                carla.Location(
                    x=spawn_transform.location.x + forward_vector.x * spawn_offset,
                    y=spawn_transform.location.y + forward_vector.y * spawn_offset,
                    z=spawn_transform.location.z
                ),
                spawn_transform.rotation
            )
            
            # Try to spawn
            slower_vehicle = self.world.try_spawn_actor(vehicle_bp, current_spawn_transform)
            if slower_vehicle is not None:
                break
                
            spawn_offset += 5.0  # Try 5 meters further on next attempt
            print(f"Spawn attempt {attempt + 1} failed, trying {spawn_offset}m further")
        
        if slower_vehicle is not None:
            self.other_vehicles.append(slower_vehicle)
            
            # Important: Don't set autopilot, we want full control
            slower_vehicle.set_autopilot(False)
            
            # Set initial velocity to target speed
            target_speed_ms = self.target_speed / 3.6  # Convert km/h to m/s
            
            # Set velocity in the direction of the road
            road_direction = spawn_transform.get_forward_vector()
            slower_vehicle.set_target_velocity(
                carla.Vector3D(
                    x=road_direction.x * target_speed_ms,
                    y=road_direction.y * target_speed_ms,
                    z=0
                )
            )
            
            # Add velocity controller to maintain slow speed
            self._setup_velocity_control(slower_vehicle)
            
            # Debug information
            actual_distance = ego_location.distance(slower_vehicle.get_location())
            print(f"Successfully spawned slower vehicle:")
            print(f"- Distance from ego: {actual_distance:.1f}m")
            print(f"- Target speed: {self.target_speed} km/h")
            print(f"- Spawn offset used: {spawn_offset}m")
            
            # Verify vehicle is in front
            relative_location = slower_vehicle.get_location() - ego_location
            angle = np.arctan2(relative_location.y, relative_location.x)
            ego_yaw = np.radians(ego_transform.rotation.yaw)
            angle_diff = np.abs(angle - ego_yaw)
            print(f"- Relative angle: {np.degrees(angle_diff):.1f} degrees")
        else:
            raise ValueError("Failed to spawn slower vehicle after multiple attempts")
    
    def _setup_velocity_control(self, vehicle):
        """Setup a velocity controller for the vehicle"""
        if not hasattr(self, '_tick_callbacks'):
            self._tick_callbacks = []
            
        def velocity_control(weak_vehicle):
            vehicle = weak_vehicle()
            if vehicle is None or not vehicle.is_alive:
                return
                
            # Get current velocity
            velocity = vehicle.get_velocity()
            speed_ms = np.sqrt(velocity.x**2 + velocity.y**2)
            target_speed_ms = self.target_speed / 3.6
            
            # PID-like control for better speed maintenance
            speed_error = target_speed_ms - speed_ms
            
            # Proportional control
            Kp = 0.5
            throttle = Kp * speed_error
            
            # Clamp throttle and add brake control
            if throttle > 0:
                throttle = min(max(throttle, 0.0), 1.0)
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(max(-throttle, 0.0), 1.0)
            
            # Get vehicle's waypoint for steering
            waypoint = self.world.get_map().get_waypoint(vehicle.get_location())
            vehicle_transform = vehicle.get_transform()
            
            # Calculate steering to stay in lane
            next_waypoint = waypoint.next(2.0)[0]
            direction = next_waypoint.transform.location - vehicle_transform.location
            forward = vehicle_transform.get_forward_vector()
            
            # Calculate steering angle
            steering = np.arctan2(direction.y, direction.x) - np.arctan2(forward.y, forward.x)
            steering = np.clip(steering, -1.0, 1.0)
            
            # Apply control
            control = carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=float(steering)
            )
            vehicle.apply_control(control)
        
        # Create weak reference to avoid circular reference
        import weakref
        weak_vehicle = weakref.ref(vehicle)
        
        # Add tick callback and store its ID
        callback_id = self.world.on_tick(lambda _: velocity_control(weak_vehicle))
        self._tick_callbacks.append(callback_id)
    
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
        
        # Get ego vehicle's waypoint for lane check
        ego_waypoint = self.world.get_map().get_waypoint(ego_transform.location)
        lead_waypoint = self.world.get_map().get_waypoint(lead_transform.location)
        
        # More comprehensive completion criteria:
        # 1. Successfully passed the slower vehicle
        # 2. Maintained safe distance
        # 3. Achieved higher speed
        # 4. Completed the overtaking maneuver (returned to original lane)
        # 5. Maintained stable control for some time
        passed_vehicle = relative_location.x < 0
        safe_distance = distance > self.safe_distance
        higher_speed = ego_speed > (self.target_speed / 3.6)
        same_lane = ego_waypoint.lane_id == lead_waypoint.lane_id
        
        return (
            passed_vehicle and
            safe_distance and
            higher_speed and
            same_lane and
            distance > self.initial_distance  # Ensure full overtake
        )
    
    def cleanup(self):
        """Clean up the scenario"""
        # Remove tick callbacks first
        if hasattr(self, '_tick_callbacks'):
            for callback_id in self._tick_callbacks:
                try:
                    self.world.remove_on_tick(callback_id)
                except:
                    pass
            self._tick_callbacks = []
        
        # Destroy all spawned vehicles
        for vehicle in self.other_vehicles:
            if vehicle is not None and vehicle.is_alive:
                try:
                    vehicle.set_autopilot(False)  # Disable autopilot before destroying
                    vehicle.destroy()
                except:
                    print(f"Warning: Failed to destroy vehicle {vehicle.id}")
        
        # Clear the vehicles list
        self.other_vehicles.clear()
        
        # Wait a tick to ensure cleanup
        self.world.tick() 