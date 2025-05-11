import carla
import numpy as np
import pygame
import math

class SensorManager:
    """
    Manages all sensors for a CARLA vehicle, including setup, data processing, and cleanup.
    This class centralizes sensor functionality that was previously scattered in the CarlaEnv class.
    """
    
    def __init__(self, world, vehicle, render_mode=None, config=None):
        """
        Initialize the sensor manager.
        
        Args:
            world: CARLA world object
            vehicle: CARLA vehicle to attach sensors to
            render_mode: Rendering mode (None, 'human', or 'rgb_array')
            config: Configuration object with sensor settings
        """
        self.world = world
        self.vehicle = vehicle
        self.render_mode = render_mode
        self.config = config
        
        # Sensor storage
        self.sensors = {}
        self.collision_data = []
        self.collision_detected = False
        self.radar_points = []
        self.camera_image = None
        
        # Collision detection parameters
        self.collision_intensity_threshold = 400.0  # Higher threshold to avoid false positives
        
        # Camera settings
        self.camera_width = 1920
        self.camera_height = 1080
        
        # Radar settings
        self.radar_max_distance = 20.0 if config is None else config.radar_range  # Maximum distance for radar observations (meters)
        self.radar_resolution = 3.0 if config is None else config.radar_resolution  # Degrees per radar observation point
        # Calculate number of radar points based on resolution (360 / resolution)
        self.radar_points_count = int(360 / self.radar_resolution)
        self.radar_points_history = {}  # Store history for temporal filtering
        
        # Setup sensors if vehicle is available
        if self.vehicle is not None:
            self.setup_sensors()
    
    def setup_sensors(self):
        """Setup all sensors for the vehicle"""
        if self.vehicle is None:
            return
            
        # Clean up existing sensors if any
        self.cleanup_sensors()
        
        blueprint_library = self.world.get_blueprint_library()
        
        # Collision sensor setup
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.sensors['collision'] = collision_sensor
        
        # Set up collision sensor callback
        collision_sensor.listen(lambda event: self._process_collision(event))
        
        # In CARLA, radar sensors don't natively support 360-degree FOV
        # We'll create multiple radar sensors to cover the full 360 degrees
        
        # Define radar parameters
        radar_range = 100.0  # Keep sensor range at 100m, but we'll filter in processing
        radar_fov = 120.0    # 120 degrees per radar (we'll use 3 radars for 360 coverage)
        
        # Create 3 radar sensors at different angles to cover 360 degrees
        radar_angles = [0, 120, 240]  # Angles in degrees
        
        for i, angle in enumerate(radar_angles):
            radar_bp = blueprint_library.find('sensor.other.radar')
            
            # Configure radar parameters
            radar_bp.set_attribute('horizontal_fov', str(radar_fov))
            radar_bp.set_attribute('vertical_fov', '20')
            radar_bp.set_attribute('range', str(radar_range))
            radar_bp.set_attribute('points_per_second', '1500')
            
            # Set the radar position and rotation
            radar_transform = carla.Transform(
                carla.Location(x=0.0, z=2.0),
                carla.Rotation(yaw=angle)  # Rotate to cover different angles
            )
            
            # Spawn the radar
            radar_sensor = self.world.spawn_actor(
                radar_bp,
                radar_transform,
                attach_to=self.vehicle
            )
            
            # Add to sensors dictionary with unique name
            self.sensors[f'radar_{i}'] = radar_sensor
            
            # Set up radar callback
            radar_sensor.listen(lambda data, radar_idx=i, radar_angle=angle: 
                               self._process_radar_data(data, radar_idx, radar_angle))
        
        # Only set up camera if rendering is enabled
        if self.render_mode:
            # Camera setup for visualization
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.camera_width))
            camera_bp.set_attribute('image_size_y', str(self.camera_height))
            camera_bp.set_attribute('fov', '90')
            
            # Set the camera position relative to the vehicle
            camera_transform = carla.Transform(
                carla.Location(x=1.6, z=1.7),  # Position slightly above and forward of the hood
                carla.Rotation(pitch=-15)       # Angle slightly downward
            )
            
            # Spawn the camera
            camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.vehicle
            )
            self.sensors['camera'] = camera
            
            # Set up camera callback
            camera.listen(lambda image: self._process_camera_data(image))
    
    def _process_collision(self, event):
        """Process collision data
        
        Args:
            event: CARLA collision event
        """
        # Get collision intensity as the magnitude of the impulse
        intensity = math.sqrt(event.normal_impulse.x**2 + 
                             event.normal_impulse.y**2 + 
                             event.normal_impulse.z**2)
        
        # Only record significant collisions
        if intensity > self.collision_intensity_threshold:
            print(f"Collision detected with intensity {intensity}")
            self.collision_data.append({
                'frame': event.frame,
                'intensity': intensity,
                'actor_id': event.other_actor.id if event.other_actor else None
            })
            self.collision_detected = True
        else:
            print(f"Minor collision ignored with intensity {intensity}")
    
    def _process_camera_data(self, image):
        """Process camera data from sensor"""
        self.camera_image = image
    
    def _process_radar_data(self, radar_data, radar_idx=0, radar_angle=0):
        """Process radar data with a simplified approach
        
        Args:
            radar_data: Raw radar data from CARLA
            radar_idx: Index of the radar sensor (0, 1, or 2)
            radar_angle: Base angle of the radar sensor in degrees
        """
        if self.vehicle is None:
            return
            
        # When a new frame comes in from a specific radar, remove outdated points from that radar
        # But don't clear all points from that radar at once - this causes jumps
        current_frame = self.world.get_snapshot().frame
        self.radar_points = [p for p in self.radar_points if 
                           p['radar_idx'] != radar_idx or 
                           (current_frame - p.get('timestamp', 0) < 5)]  # Keep recent points
        
        # Convert radar angle to radians
        radar_angle_rad = np.radians(radar_angle)
        
        # Temporary list to store new points from this radar update
        new_radar_points = []
        
        # Process each radar detection
        for detection in radar_data:
            # More aggressive filtering for noise reduction
            
            # 1. Filter out points that are too close or too far
            if detection.depth < 0.5 or detection.depth > self.radar_max_distance:
                continue
                
            # 2. Filter out points with low intensity (increased threshold)
            if hasattr(detection, 'intensity') and detection.intensity < 0.3:  # More aggressive filtering
                continue
                
            # 3. Calculate adjusted azimuth (angle)
            adjusted_azimuth = detection.azimuth + radar_angle_rad
            
            # 4. Calculate x, y coordinates (in vehicle's frame)
            x = detection.depth * np.cos(detection.altitude) * np.cos(adjusted_azimuth)
            y = detection.depth * np.cos(detection.altitude) * np.sin(adjusted_azimuth)
            
            # 5. Skip points with invalid coordinates
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                continue
                
            # 6. Filter out points with extreme altitude angles (likely ground or sky reflections)
            if abs(np.degrees(detection.altitude)) > 8:  # Stricter altitude filtering
                continue
                
            # 7. Create a unique key for this approximate spatial location (rounded to 1.0m grid for broader grouping)
            grid_size = 1.0  # Increased grid size for better clustering
            location_key = f"{int(x/grid_size)},{int(y/grid_size)}"
            
            # 8. Store essential data
            point = {
                'x': float(x),  # Ensure it's a regular float, not numpy float
                'y': float(y),
                'velocity': float(detection.velocity),
                'radar_idx': int(radar_idx),
                'location_key': location_key,
                'timestamp': current_frame  # Track when this point was last seen
            }
            
            # Add to the temporary list of new points
            new_radar_points.append(point)
            
            # Update the history for this location - with exponential moving average for stability
            if location_key not in self.radar_points_history:
                self.radar_points_history[location_key] = {
                    'points': [point],
                    'count': 1,
                    'last_seen': current_frame,
                    'avg_x': x,
                    'avg_y': y,
                    'avg_velocity': detection.velocity
                }
            else:
                history = self.radar_points_history[location_key]
                
                # Update exponential moving averages (EMA) with alpha=0.3
                # This gives more weight to historical values, making it smoother
                alpha = 0.3
                history['avg_x'] = alpha * x + (1 - alpha) * history['avg_x']
                history['avg_y'] = alpha * y + (1 - alpha) * history['avg_y']
                history['avg_velocity'] = alpha * detection.velocity + (1 - alpha) * history['avg_velocity']
                
                # Update history
                history['points'].append(point)
                history['count'] += 1
                history['last_seen'] = current_frame
                
                # Keep only the most recent observations for this location
                if len(history['points']) > 10:  # Track more history points
                    history['points'] = history['points'][-10:]
        
        # Apply smarter temporal filtering with different levels of stability
        stable_points = []
        
        for point in new_radar_points:
            history = self.radar_points_history[point['location_key']]
            
            # Higher count = more confidence = more stability
            stability_level = min(1.0, history['count'] / 10.0)  # Normalized to 0.0-1.0
            
            # Create a point with smoothed position
            stable_point = {
                'x': float(history['avg_x']),  # Use the EMA for smoother tracking
                'y': float(history['avg_y']),
                'velocity': float(history['avg_velocity']),
                'radar_idx': point['radar_idx'],
                'timestamp': current_frame,
                'is_stable': history['count'] >= 3,  # Boolean flag for stability
                'stability': stability_level,  # Continuous stability measure
                'location_key': point['location_key']
            }
            
            # Add the stable point - all points get added but with stability measure
            stable_points.append(stable_point)
        
        # Add the stable points to our main radar points list
        self.radar_points.extend(stable_points)
        
        # Clean up old history entries (not seen recently)
        keys_to_remove = []
        
        # Use a copy of the dictionary keys to avoid the "dictionary changed size during iteration" error
        for key in list(self.radar_points_history.keys()):
            history = self.radar_points_history[key]
            if current_frame - history['last_seen'] > 20:  # Increased persistence
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.radar_points_history[key]
        
        # Limit the number of points to prevent memory issues
        if len(self.radar_points) > 1000:
            self.radar_points = self.radar_points[-1000:]
    
    def get_radar_observation(self):
        """Get a simplified observation from radar data
        
        Returns:
            numpy.ndarray: Array containing distance measurements at each angle
        """
        # Create a simple array to hold distance measurements
        # Shape is determined by radar resolution
        radar_obs = np.full(self.radar_points_count, self.radar_max_distance)  # Fill with max range
        
        # If no radar data is available, return the default observation
        if not self.radar_points:
            return radar_obs.reshape(1, self.radar_points_count)  # Reshape to match expected dimensions
        
        # First pass: collect all points by angle
        angle_data = {}
        
        for point in self.radar_points:
            # Calculate angle and distance
            x, y = point['x'], point['y']
            
            # Skip if x or y is NaN
            if np.isnan(x) or np.isnan(y):
                continue
                
            # Calculate angle in degrees (0-360)
            angle_rad = np.arctan2(y, x)
            angle_deg = np.degrees(angle_rad)
            angle_deg = (angle_deg + 360) % 360
            
            # Calculate distance
            distance = np.sqrt(x*x + y*y)
            
            # Skip points beyond our observation distance
            if distance > self.radar_max_distance:
                continue
                
            # Calculate angle index based on radar resolution
            angle_idx = int(angle_deg / self.radar_resolution)
            if angle_idx >= self.radar_points_count:  # Handle edge case
                angle_idx = self.radar_points_count - 1
            
            # Store each point with its stability and distance
            if angle_idx not in angle_data:
                angle_data[angle_idx] = []
            
            stability = point.get('stability', 0.0)
            is_stable = point.get('is_stable', False)
            
            angle_data[angle_idx].append({
                'distance': distance,
                'stability': stability,
                'is_stable': is_stable
            })
        
        # Second pass: for each angle, select the best point based on stability and distance
        for angle_idx, points in angle_data.items():
            if not points:
                continue
                
            # Sort by stability (descending) and then by distance (ascending)
            points.sort(key=lambda p: (-p['stability'], p['distance']))
            
            # Take the best point (most stable, or closest if equally stable)
            best_point = points[0]
            radar_obs[angle_idx] = best_point['distance']
        
        # Apply Gaussian smoothing to the radar observation to reduce noise
        # Create a circular kernel for proper handling of the angle wrap-around
        kernel_size = min(3, max(1, int(5 / self.radar_resolution)))  # Adjust kernel size based on resolution
        sigma = 1.0
        
        # Create a copy for smoothing (to handle the circular nature of angles)
        extended_obs = np.concatenate([radar_obs[-kernel_size:], radar_obs, radar_obs[:kernel_size]])
        
        # Create Gaussian kernel
        kernel = np.exp(-np.linspace(-kernel_size, kernel_size, 2*kernel_size+1)**2 / (2*sigma**2))
        kernel = kernel / np.sum(kernel)  # Normalize
        
        # Apply the filter
        smoothed_extended = np.convolve(extended_obs, kernel, mode='same')
        
        # Extract the original part and handle special cases (max distance)
        # Only smooth points that are not at max distance
        for i in range(self.radar_points_count):
            idx = i + kernel_size
            if radar_obs[i] < self.radar_max_distance:
                radar_obs[i] = smoothed_extended[idx]
        
        # Reshape to match the expected dimensions (1 layer, radar_points_count angles)
        return radar_obs.reshape(1, self.radar_points_count)
    
    def render_radar_visualization(self, screen, camera_width, camera_height):
        """Render radar data visualization on the Pygame screen"""
        if screen is None:
            return
            
        # Define visualization parameters
        radar_surface_width = 200
        radar_surface_height = 200
        radar_surface = pygame.Surface((radar_surface_width, radar_surface_height))
        radar_surface.fill((0, 0, 0))  # Black background
        radar_surface.set_alpha(220)  # Slight transparency
        
        # Draw a border around the radar visualization
        pygame.draw.rect(radar_surface, (100, 100, 100), pygame.Rect(0, 0, radar_surface_width, radar_surface_height), 1)
        
        # Draw center point (ego vehicle position)
        center_x = radar_surface_width // 2
        center_y = radar_surface_height // 2
        pygame.draw.circle(radar_surface, (255, 255, 255), (center_x, center_y), 3)
        
        # Draw range circles - adjusted to show the configured max distance
        for range_val in [10, 20, self.radar_max_distance]:
            radius = int((range_val / self.radar_max_distance) * (min(center_x, center_y) - 10))
            pygame.draw.circle(radar_surface, (50, 50, 50), (center_x, center_y), radius, 1)
        
        # Draw cardinal directions - rotated so vehicle front faces north
        font = pygame.font.SysFont('Arial', 10)
        # North (vehicle front)
        text = font.render('N', True, (200, 200, 200))
        radar_surface.blit(text, (center_x - text.get_width() // 2, 5))
        # East (vehicle right)
        text = font.render('E', True, (200, 200, 200))
        radar_surface.blit(text, (radar_surface_width - 15, center_y - text.get_height() // 2))
        # South (vehicle rear)
        text = font.render('S', True, (200, 200, 200))
        radar_surface.blit(text, (center_x - text.get_width() // 2, radar_surface_height - 15))
        # West (vehicle left)
        text = font.render('W', True, (200, 200, 200))
        radar_surface.blit(text, (5, center_y - text.get_height() // 2))
        
        # Draw radar points - scale based on configured max distance
        scale_factor = (min(center_x, center_y) - 10) / self.radar_max_distance
        
        # Draw radar points
        if self.radar_points:
            # Only draw points within the max distance
            visible_points = [p for p in self.radar_points if 
                             np.sqrt(p['x']**2 + p['y']**2) <= self.radar_max_distance]
            
            # Count stable vs unstable points
            stable_count = sum(1 for p in visible_points if 'is_stable' in p and p['is_stable'])
            
            for point in visible_points:
                # Skip points with NaN values
                if np.isnan(point['x']) or np.isnan(point['y']):
                    continue
                    
                # Rotate coordinates so vehicle front faces north
                # In the original coordinate system: x is forward, y is right
                # In the rotated system: x is right, y is backward (to match screen coordinates)
                rotated_x = point['y']  # Original y becomes x (right)
                rotated_y = -point['x']  # Negative original x becomes y (down)
                
                # Convert to display coordinates
                x = center_x + int(rotated_x * scale_factor)
                y = center_y + int(rotated_y * scale_factor)
                
                # Check if point is within display bounds
                if 0 <= x < radar_surface_width and 0 <= y < radar_surface_height:
                    # All points are green, but stable points are brighter
                    if 'is_stable' in point and point['is_stable']:
                        color = (0, 255, 0)  # Bright green for stable points
                    else:
                        color = (0, 150, 0)  # Darker green for unstable points
                    
                    # Size still varies with velocity for approaching objects
                    velocity = point['velocity']
                    point_size = 2
                    if velocity < 0:  # Approaching
                        point_size = min(5, 2 + int(abs(velocity) / 5))
                    
                    pygame.draw.circle(radar_surface, color, (x, y), point_size)
            
            # Add point count to the visualization
            count_text = font.render(f"Points: {len(visible_points)} (Stable: {stable_count})", True, (200, 200, 200))
            radar_surface.blit(count_text, (5, radar_surface_height - 20))
        
        # Add a title with max distance and resolution
        title_font = pygame.font.SysFont('Arial', 12)
        title_text = title_font.render(f'Radar View ({self.radar_max_distance}m, {self.radar_resolution}° res)', True, (255, 255, 255))
        radar_surface.blit(title_text, (center_x - title_text.get_width() // 2, 5))
        
        # Position the radar visualization in the bottom-right corner of the camera view
        screen.blit(radar_surface, (camera_width - radar_surface_width - 10, 
                                   camera_height - radar_surface_height - 10))
    
    def get_camera_image_array(self):
        """Convert camera image to numpy array for rendering"""
        if self.camera_image is None:
            return None
            
        # Convert CARLA image to numpy array properly
        array = np.frombuffer(self.camera_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.camera_image.height, self.camera_image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # Convert from BGR to RGB
        
        return array
    
    def reset(self):
        """Reset the sensor data"""
        self.collision_data = []
        self.collision_detected = False
        self.radar_points = []
        self.camera_image = None
    
    def cleanup_sensors(self):
        """Clean up all sensors"""
        for sensor in self.sensors.values():
            if sensor and sensor.is_alive:
                sensor.destroy()
        self.sensors = {}
    
    def set_vehicle(self, vehicle):
        """Update the vehicle and reattach sensors"""
        self.vehicle = vehicle
        if vehicle is not None:
            self.setup_sensors()
    
    def set_radar_max_distance(self, distance):
        """Update the maximum radar distance"""
        self.radar_max_distance = distance 