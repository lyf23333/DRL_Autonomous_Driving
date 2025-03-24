import carla
import gymnasium as gym
import numpy as np
import pygame
import math

from ..mdp.observation_manager import ObservationManager
from ..mdp.action_manager import ActionManager
from ..utils.env_utils import spawn_ego_vehicle, generate_random_waypoints, check_decision_points
from ..mdp.rewards import calculate_reward
from ..utils.viz_utils import render_trust_visualization, render_waypoints_on_camera
from ..utils.sensors import SensorManager
from ..utils.termination_manager import TerminationManager
from ..trust.trust_interface import TrustInterface

class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gymnasium interface"""
    
    def __init__(self, trust_interface, config):
        self._initialized = False
        super(CarlaEnv, self).__init__()
        
        # Load configuration
        self.config = config
        
        # Connect to CARLA server
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.get_world()
        self.render_mode = self.config.render_mode

        # Spawn new vehicle
        self.vehicle, self.spawn_point = spawn_ego_vehicle(self.world)

        # Initialize observation and action managers
        self.sensor_manager = SensorManager(self.world, self.vehicle, self.render_mode)
        self.observation_manager = ObservationManager(self.config, self.sensor_manager)
        self.action_manager = ActionManager(self.config)
        
        # Set the action and observation spaces from the managers
        self.action_space = self.action_manager.action_space
        self.observation_space = self.observation_manager.observation_space
        
        # Scenario management
        self.active_scenario = None
        
        # Trust-related attributes
        self.trust_interface: TrustInterface = trust_interface

        # Environment attributes
        self.last_step_time = None
        self.step_count = 0
        self.max_episode_steps = self.config.max_episode_steps
        
        # Target speed attributes
        self.base_target_speed = self.config.base_target_speed  # km/h at max trust
        self.min_target_speed = self.config.min_target_speed    # km/h at min trust
        self.target_speed = self.base_target_speed  # Default to base speed
        
        # Previous control state for detecting changes
        self.prev_control = None
        
        # Camera view setup
        self.camera_width = self.config.camera_width
        self.camera_height = self.config.camera_height
        self.pygame_initialized = False
        self.screen = None
        
        # Trust visualization
        self.trust_history = []
        self.max_trust_history = self.config.max_trust_history
        self.trust_viz_height = self.config.trust_viz_height
        
        # Reward visualization
        self.reward_history = []
        self.max_reward_history = self.config.max_reward_history
        self.episode_reward = 0.0      # Cumulative reward for current episode
        
        # Reward component tracking
        self.reward_components = {
            'path': 0.0,
            'progress': 0.0,
            'safety': 0.0,
            'comfort': 0.0,
            'trust': 0.0,
            'intervention': 0.0
        }
        
        # Only initialize pygame if render_mode is True
        if self.render_mode:
            try:
                pygame.init()
                # Create a taller screen to accommodate trust visualization
                self.screen = pygame.display.set_mode((self.camera_width, self.camera_height + self.trust_viz_height))
                pygame.display.set_caption("CARLA Environment - Vehicle View")
                self.pygame_initialized = True
                # Initialize font for text rendering
                self.font = pygame.font.SysFont('Arial', 16)
            except:
                print("Warning: Pygame initialization failed. Camera view will not be displayed.")
                self.pygame_initialized = False
        
        # Path following attributes
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = self.config.waypoint_threshold  # meters
        self.path_length = self.config.path_length  # Number of waypoints to generate
        
        # Generate initial random waypoints if vehicle is spawned
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world)
        
        # Visualization settings
        self.show_waypoints = self.config.show_waypoints  # Flag to toggle waypoint visualization
        self.waypoint_lookahead = self.config.waypoint_lookahead  # Number of waypoints to show ahead
        
        # Initialize termination manager
        self.termination_manager = TerminationManager(max_episode_steps=self.max_episode_steps)
        
    def set_scenario(self, scenario, config=None):
        """Set the active scenario for the environment"""
        self.active_scenario = scenario
    
    def set_waypoints(self, waypoints):
        """Set the waypoints for path following"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
    
    def step(self, action):
        """Take a step in the environment"""
        if not hasattr(self, 'vehicle') or self.vehicle is None:
            # Return empty observation, zero reward, and terminal state if vehicle doesn't exist
            obs = self.observation_manager.get_observation(
                None, [], 0, self.waypoint_threshold, 
                self.trust_interface, self.active_scenario,
                target_speed=self.target_speed
            )
            return obs, 0.0, True, False, {}
            
        # Store previous control for comparison
        if self.prev_control is None:
            self.prev_control = self.vehicle.get_control()

        # Check for decision points (intersections, lane merges, etc.)
        is_near_decision_point = check_decision_points(self.vehicle, self.world, self.config.decision_point_distance)
        self.trust_interface.set_near_decision_point(is_near_decision_point)
            
        # Process action through action manager - with trust-based adjustments
        adjusted_action, current_intervention_prob = self.action_manager.adjust_action_with_trust(
            action, self.trust_interface, is_near_decision_point, is_adjusting=False
        )
        
        # Generate CARLA vehicle control from processed action
        control = self.action_manager.generate_vehicle_control(adjusted_action)
        
        # Apply control to vehicle
        self.vehicle.apply_control(control)
        
        # Update the world
        self.world.tick()
        
        # Update the observation manager with current vehicle state
        self.observation_manager.update(self.vehicle)
        
        # Check if we've reached the current waypoint and update if needed
        self._update_waypoint_index()
        
        # Render if needed (but don't break training if rendering fails)
        if self.render_mode:
            try:
                self.render()
            except Exception as e:
                print(f"Warning: Rendering failed: {e}")
        
        # Update driving metrics in trust interface
        self.trust_interface.update_driving_metrics(self.vehicle)
        
        # Calculate time delta since last step
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        dt = current_time - self.last_step_time if self.last_step_time is not None else 0.0
        self.last_step_time = current_time
        
        # Detect manual interventions based on control changes and update trust
        self.trust_interface.detect_interventions_and_update_trust(
            control, 
            self.prev_control, 
            self.world.get_snapshot(),
            dt=dt
        )
        
        # Update trust-based behavior parameters in trust interface
        self._update_target_speed()
        self.trust_interface.update_behavior_adjustment()
        
        # Update trust history for visualization
        if len(self.trust_history) >= self.max_trust_history:
            self.trust_history.pop(0)
        self.trust_history.append(self.trust_interface.trust_level)
        
        # Calculate reward
        reward, self.reward_components = calculate_reward(
            self.vehicle, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.waypoint_threshold, 
            self.trust_interface, 
            self.active_scenario, 
            self.world, 
            self.target_speed,
            reward_weights=self.config.reward_weights
        )
        
        # Update reward history for visualization
        self.episode_reward += reward
        if len(self.reward_history) >= self.max_reward_history:
            self.reward_history.pop(0)
        self.reward_history.append(reward)
        
        # Check if done
        terminated, truncated = self.termination_manager.check_termination(
            self.vehicle, 
            self.world, 
            self.step_count, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.active_scenario, 
            self.sensor_manager.collision_detected
        )
        
        # Get observation using observation manager
        obs = self.observation_manager.get_observation(
            self.vehicle, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.waypoint_threshold, 
            self.trust_interface, 
            self.active_scenario,
            target_speed=self.target_speed
        )
        
        # Store current control for next comparison
        self.prev_control = control
        
        # Additional info
        info = {
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.75,
            'current_speed': 3.6 * np.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2) if self.vehicle else 0.0,
            'target_speed': self.target_speed,
            'step_count': self.step_count,
            'driving_metrics': self.trust_interface.driving_metrics if self.trust_interface else {},
            'is_near_decision_point': is_near_decision_point,
            'behavior_adjustment': self.trust_interface.behavior_adjustment,
            'intervention_probability': current_intervention_prob,
            'intervention_active': self.trust_interface.intervention_active
        }

        self.step_count += 1
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reset the environment
        
        Args:
            seed: The seed for random number generation
            options: Additional options for environment reset
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset step counter
        self.step_count = 0
        
        # Reset action manager
        self.action_manager.reset()
        
        # Reset observation manager
        self.observation_manager.reset()
        
        # Reset trust history
        self.trust_history = []
        
        # Reset reward history
        self.reward_history = []
        self.episode_reward = 0.0
        
        # Reset reward components
        self.reward_components = {
            'path': 0.0,
            'progress': 0.0,
            'safety': 0.0,
            'comfort': 0.0,
            'trust': 0.0,
            'intervention': 0.0
        }
        
        # Reset trust-related attributes
        self.prev_control = None
        self.last_step_time = None
        self.trust_interface.reset()
        
        # Reset termination manager
        self.termination_manager.reset()
            
        # Destroy existing vehicle if any
        if self.vehicle.is_alive:
            self.vehicle.destroy()

        # Spawn new vehicle
        self.vehicle, self.spawn_point = spawn_ego_vehicle(self.world)
        
        # Setup sensor manager with the new vehicle
        self.sensor_manager.reset()
        self.sensor_manager.set_vehicle(self.vehicle)
        
        # Generate random waypoints for the new vehicle position
        self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world)
        
        # Setup active scenario if exists
        if self.active_scenario and not self.active_scenario.is_setup:
            self.active_scenario.setup()
        
        # Tick the world to update
        self.world.tick()
        
        # Initialize the observation manager with the vehicle's initial position
        self.observation_manager.update(self.vehicle)
        
        # Get observation using observation manager
        obs = self.observation_manager.get_observation(
            self.vehicle, 
            self.waypoints, 
            self.current_waypoint_idx, 
            self.waypoint_threshold, 
            self.trust_interface, 
            self.active_scenario,
            target_speed=self.target_speed
        )
        
        # Additional info
        info = {
            'spawn_point': f"({self.spawn_point.location.x:.1f}, {self.spawn_point.location.y:.1f}, {self.spawn_point.location.z:.1f})",
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.75,
            'target_speed': self.target_speed
        }
        
        return obs, info
    
    def _update_target_speed(self):
        """Update vehicle behavior parameters based on trust level and driving metrics"""
        # Get current trust level
        trust_level = self.trust_interface.trust_level
        
        # 1. Update target speed based on trust level
        self.target_speed = self.min_target_speed + (self.base_target_speed - self.min_target_speed) * trust_level
    
    def close(self):
        """Clean up resources when environment is closed"""
        
        # Clean up sensors
        if self.sensor_manager:
            self.sensor_manager.cleanup_sensors()
        
        # Clean up vehicle
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
        
        # Clean up scenario
        if self.active_scenario:
            self.active_scenario.cleanup()

        # Clean up trust interface
            self.trust_interface.cleanup()
            
        # Clean up pygame
        if hasattr(self, 'pygame_initialized') and self.pygame_initialized:
            try:
                pygame.quit()
                self.pygame_initialized = False
            except:
                pass

    def render(self):
        """Render the environment"""
        if self.render_mode is None or not self.pygame_initialized:
            return
            
        # Get camera image from sensor manager
        if self.sensor_manager:
            camera_array = self.sensor_manager.get_camera_image_array()
            
            if camera_array is not None:
                # Convert to pygame surface
                pygame_image = pygame.surfarray.make_surface(camera_array.swapaxes(0, 1))
                
                # Display the image
                self.screen.blit(pygame_image, (0, 0))
                
                # Draw waypoints on camera view if enabled
                if self.show_waypoints and self.waypoints and hasattr(self, 'vehicle') and self.vehicle:
                    render_waypoints_on_camera(
                        self.screen, 
                        self.sensor_manager.sensors, 
                        self.camera_width, 
                        self.camera_height, 
                        self.waypoints, 
                        self.current_waypoint_idx, self.waypoint_lookahead
                    )
                
                # Draw trust visualization panel
                if hasattr(self, 'font'):
                    render_trust_visualization(
                        self.screen, 
                        self.font, 
                        self.trust_interface, 
                        self.vehicle, 
                        self.camera_width, 
                        self.camera_height, 
                        self.trust_viz_height, 
                        self.reward_components, 
                        self.trust_history, 
                        self.max_trust_history,
                        self.target_speed
                    )
                
                # Render radar visualization
                self.sensor_manager.render_radar_visualization(
                    self.screen, 
                    self.camera_width, 
                    self.camera_height
                )
                
                # Update the display
                pygame.display.flip()
                
                # Process pygame events to keep the window responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        self.pygame_initialized = False
                
                return camera_array  # Return the RGB array for 'rgb_array' mode
        
        return None

    def _update_waypoint_index(self):
        """
        Check if we've reached the current waypoint and update the waypoint index accordingly.
        """
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints) or self.vehicle is None:
            return
            
        # Get vehicle location
        vehicle_location = self.vehicle.get_location()
        
        # Get current waypoint location
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        waypoint_location = current_waypoint.transform.location
        
        # Calculate distance to current waypoint
        distance = math.sqrt(
            (vehicle_location.x - waypoint_location.x) ** 2 +
            (vehicle_location.y - waypoint_location.y) ** 2
        )
        
        # If we've reached the waypoint, increment the index
        if distance < self.waypoint_threshold:
            self.current_waypoint_idx += 1
            
            # Log waypoint progress
            waypoints_remaining = len(self.waypoints) - self.current_waypoint_idx
            if waypoints_remaining > 0:
                print(f"Reached waypoint {self.current_waypoint_idx}/{len(self.waypoints)}. {waypoints_remaining} remaining.")
            else:
                print(f"Reached final waypoint! ({self.current_waypoint_idx}/{len(self.waypoints)})")
                
            # Limit the index to the available waypoints
            self.current_waypoint_idx = min(self.current_waypoint_idx, len(self.waypoints) - 1)

        if self.current_waypoint_idx >= len(self.waypoints) - 1:
            return
        # Get current waypoint location
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        next_waypoint = self.waypoints[self.current_waypoint_idx + 1]

        vehicle_to_current_waypoint_vector = np.array([
            current_waypoint.transform.location.x - vehicle_location.x,
            current_waypoint.transform.location.y - vehicle_location.y
        ])

        current_waypoint_to_next_waypoint_direction_vector = np.array([
            next_waypoint.transform.location.x - current_waypoint.transform.location.x,
            next_waypoint.transform.location.y - current_waypoint.transform.location.y
        ])

        
        # Normalize vectors
        vehicle_to_current_waypoint_vector = vehicle_to_current_waypoint_vector / np.linalg.norm(vehicle_to_current_waypoint_vector)
        current_waypoint_to_next_waypoint_direction_vector = current_waypoint_to_next_waypoint_direction_vector / np.linalg.norm(current_waypoint_to_next_waypoint_direction_vector)
        dot_product = np.dot(vehicle_to_current_waypoint_vector, current_waypoint_to_next_waypoint_direction_vector)
        
        angle = np.arccos(dot_product)

        if angle > math.pi / 2:
            self.current_waypoint_idx += 1
            print(f"Waypoint skipped due to large angle ({math.degrees(angle):.1f}° > {math.degrees(math.pi / 2):.1f}°)")
            
            # Limit the index to the available waypoints
            self.current_waypoint_idx = min(self.current_waypoint_idx, len(self.waypoints) - 1)