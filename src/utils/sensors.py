import carla
import numpy as np
import pygame
import math

class SensorManager:
    """
    Manages all sensors for a CARLA vehicle, including setup, data processing, and cleanup.
    This class centralizes sensor functionality that was previously scattered in the CarlaEnv class.
    """
    
    def __init__(self, world, vehicle, render_mode=None):
        """
        Initialize the sensor manager.
        
        Args:
            world: CARLA world object
            vehicle: CARLA vehicle to attach sensors to
            render_mode: Rendering mode (None, 'human', or 'rgb_array')
        """
        self.world = world
        self.vehicle = vehicle
        self.render_mode = render_mode
        
        # Sensor storage
        self.sensors = {}
        
        # Camera settings
        self.camera_width = 800
        self.camera_height = 600
        self.camera_image = None
        
        # Radar settings
        self.radar_max_distance = 20.0  # Maximum distance for radar observations (meters)
        self.radar_points = []  # Store current radar points
        self.radar_points_history = {}  # Store history for temporal filtering
        
        # Collision data
        self.collision_detected = False
        self.collision_impulse = np.zeros(3, dtype=np.float32)
        
        # Setup sensors if vehicle is available
        if self.vehicle is not None:
            self.setup_sensors()
    
    def setup_sensors(self):
        """Setup all sensors for the vehicle"""
        if self.vehicle is None:
            return
            
        # Clean up existing sensors if any
        self._cleanup_sensors()
        
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
        """Process collision events"""
        # Get the impulse from the collision
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        # Only register significant collisions
        if intensity > 0.5:  # Threshold to filter out minor collisions
            self.collision_detected = True
            self.collision_impulse = np.array([impulse.x, impulse.y, impulse.z], dtype=np.float32)
    
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
            
        # When a new frame comes in from a specific radar, clear previous points from that radar
        # Filter out points from other radars
        self.radar_points = [p for p in self.radar_points if p['radar_idx'] != radar_idx]
        
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
            if hasattr(detection, 'intensity') and detection.intensity < 0.2:  # Increased from 0.1 to 0.2
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
            if abs(np.degrees(detection.altitude)) > 10:  # Filter points with altitude > 10 degrees
                continue
                
            # 7. Create a unique key for this approximate spatial location (rounded to 0.5m grid)
            # This helps track the same physical object across frames
            grid_size = 0.5  # 0.5 meter grid
            location_key = f"{int(x/grid_size)},{int(y/grid_size)}"
            
            # 8. Store essential data
            point = {
                'x': float(x),  # Ensure it's a regular float, not numpy float
                'y': float(y),
                'velocity': float(detection.velocity),
                'radar_idx': int(radar_idx),
                'location_key': location_key,
                'timestamp': self.world.get_snapshot().frame  # Track when this point was last seen
            }
            
            # Add to the temporary list of new points
            new_radar_points.append(point)
            
            # Update the history for this location
            if location_key not in self.radar_points_history:
                self.radar_points_history[location_key] = {
                    'points': [point],
                    'count': 1,
                    'last_seen': self.world.get_snapshot().frame
                }
            else:
                history = self.radar_points_history[location_key]
                history['points'].append(point)
                history['count'] += 1
                history['last_seen'] = self.world.get_snapshot().frame
                
                # Keep only the last 5 observations for this location
                if len(history['points']) > 5:
                    history['points'] = history['points'][-5:]
        
        # Apply temporal filtering - only keep points that have been seen multiple times
        # or are very recent (just detected)
        stable_points = []
        for point in new_radar_points:
            history = self.radar_points_history[point['location_key']]
            
            # Accept points that have been seen multiple times (stable)
            if history['count'] >= 3:
                # Use averaged position from history for more stability
                recent_points = history['points'][-3:]  # Last 3 observations
                avg_x = sum(p['x'] for p in recent_points) / len(recent_points)
                avg_y = sum(p['y'] for p in recent_points) / len(recent_points)
                avg_velocity = sum(p['velocity'] for p in recent_points) / len(recent_points)
                
                # Create a stabilized point
                stable_point = {
                    'x': float(avg_x),
                    'y': float(avg_y),
                    'velocity': float(avg_velocity),
                    'radar_idx': point['radar_idx'],
                    'is_stable': True  # Mark as a stable point
                }
                stable_points.append(stable_point)
            # Also accept very new points (first few detections)
            elif history['count'] <= 2:
                point['is_stable'] = False  # Mark as not yet stable
                stable_points.append(point)
        
        # Add the stable points to our main radar points list
        self.radar_points.extend(stable_points)
        
        # Clean up old history entries (not seen recently)
        current_frame = self.world.get_snapshot().frame
        keys_to_remove = []
        for key, history in self.radar_points_history.items():
            if current_frame - history['last_seen'] > 10:  # Not seen for 10 frames
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
        # Shape: [360 azimuth angles] - we'll use just one layer for simplicity
        radar_obs = np.full(360, self.radar_max_distance)  # Fill with max range
        
        # If no radar data is available, return the default observation
        if not self.radar_points:
            return radar_obs.reshape(1, 360)  # Reshape to match expected dimensions
            
        # Process each point in the radar data
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
                
            # Calculate angle index (1-degree resolution)
            angle_idx = int(angle_deg)
            if angle_idx >= 360:  # Handle edge case
                angle_idx = 359
                
            # Prioritize stable points and closer points
            if radar_obs[angle_idx] > distance or ('is_stable' in point and point['is_stable']):
                radar_obs[angle_idx] = distance
        
        # Reshape to match the expected dimensions (1 layer, 360 angles)
        return radar_obs.reshape(1, 360)
    
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
        title_text = title_font.render(f'Radar View ({self.radar_max_distance}m, 1° res)', True, (255, 255, 255))
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
        """Reset sensor data"""
        self.collision_detected = False
        self.collision_impulse = np.zeros(3, dtype=np.float32)
        self.radar_points = []
        self.radar_points_history = {}
        self.camera_image = None
    
    def _cleanup_sensors(self):
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