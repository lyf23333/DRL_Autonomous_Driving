import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import pygame

from ..utils.env_utils import generate_control_from_action, spawn_ego_vehicle, generate_random_waypoints, check_decision_points
from ..mdp.observation import get_obs
from ..mdp.rewards import calculate_reward
from ..utils.viz_utils import render_trust_visualization, render_waypoints_on_camera
from ..utils.sensors import SensorManager
from ..utils.termination_manager import TerminationManager
from ..trust.trust_interface import TrustInterface

class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gymnasium interface"""
    
    def __init__(self, trust_interface, town='Town01', port=2000, render_mode=None):
        self._initialized = False
        super(CarlaEnv, self).__init__()
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.step_count = 0
        self.max_episode_steps = 1000
        
        # Scenario management
        self.active_scenario = None
        
        # Trust-related attributes
        self.trust_interface: TrustInterface = trust_interface
        self.last_step_time = None
        self.current_intervention_prob = 0.0  # Probability of intervention in current step
        
        # Decision point detection
        self.is_near_decision_point = False
        self.decision_point_distance = 20.0  # meters
        
        # Target speed attributes
        self.base_target_speed = 20.0  # km/h at max trust
        self.min_target_speed = 5.0    # km/h at min trust
        self.target_speed = self.base_target_speed  # Default to base speed
        
        # Previous control state for detecting changes
        self.prev_control = None
        
        # Initialize sensor manager (will be properly set up when vehicle is spawned)
        self.sensor_manager = None
        self.render_mode = render_mode
        
        # Camera view setup
        self.camera_width = 800
        self.camera_height = 600
        self.pygame_initialized = False
        self.screen = None
        
        # Trust visualization
        self.trust_history = []
        self.max_trust_history = 100  # Number of trust values to keep in history
        self.trust_viz_height = 280  # Increased from 220 to accommodate more information
        
        # Reward visualization
        self.reward_history = []
        self.max_reward_history = 100  # Number of reward values to keep in history
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
        
        # Set up action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # [steering, throttle/brake]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space includes vehicle state, path info, and intervention history
        self.observation_space = spaces.Dict({
            'vehicle_state': spaces.Box(
                low=np.array([-np.inf] * 16),
                high=np.array([np.inf] * 16),
                dtype=np.float32
            ),
            'recent_intervention': spaces.Discrete(2),  # Binary: 0 or 1
            'scenario_obs': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(15,),  # Updated for 3 vehicles with 5 values each
                dtype=np.float32
            ),
            'radar_obs': spaces.Box(
                low=0.0,
                high=20.0,  # Maximum radar range is now configurable
                shape=(1, 360),  # 1 layer, 360 azimuth angles (1-degree resolution)
                dtype=np.float32
            )
        })
        
        # Path following attributes
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 2.0  # meters
        self.path_length = 20  # Number of waypoints to generate
        
        # Generate initial random waypoints if vehicle is spawned
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world)
        
        # Visualization settings
        self.show_waypoints = True  # Flag to toggle waypoint visualization
        self.waypoint_lookahead = 20  # Number of waypoints to show ahead
        
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
            return self._get_obs(), 0.0, True, False, {}
            
        # Store previous control for comparison
        if self.prev_control is None:
            self.prev_control = self.vehicle.get_control()
            
        # Apply action to vehicle - with trust-based adjustments
        adjusted_action = self._adjust_action_based_on_trust(action)
        control = generate_control_from_action(adjusted_action)
        self.vehicle.apply_control(control)
        
        # Update the world
        self.world.tick()
        
        # Render if needed (but don't break training if rendering fails)
        if self.render_mode:
            try:
                self.render()
            except Exception as e:
                print(f"Warning: Rendering failed: {e}")
        
        # Check for decision points (intersections, lane merges, etc.)
        self.is_near_decision_point = check_decision_points(self.vehicle, self.world, self.decision_point_distance)
        self.trust_interface.set_near_decision_point(self.is_near_decision_point)

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
            self.target_speed
        )
        
        # Update reward history for visualization
        self.episode_reward += reward
        if len(self.reward_history) >= self.max_reward_history:
            self.reward_history.pop(0)
        self.reward_history.append(reward)
        
        self.step_count += 1
        
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
        
        # Get observation
        obs = get_obs(self.vehicle, self.waypoints, self.current_waypoint_idx, self.waypoint_threshold, self.trust_interface, self.active_scenario)
        
        # Add radar observation to the observation dictionary
        obs['radar_obs'] = self.sensor_manager.get_radar_observation()
        
        # Store current control for next comparison
        self.prev_control = control
        
        # Additional info
        info = {
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.75,
            'current_speed': 3.6 * np.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2) if self.vehicle else 0.0,
            'target_speed': self.target_speed,
            'step_count': self.step_count,
            'driving_metrics': self.trust_interface.driving_metrics if self.trust_interface else {},
            'is_near_decision_point': self.is_near_decision_point,
            'behavior_adjustment': self.trust_interface.behavior_adjustment,
            'intervention_probability': self.current_intervention_prob,
            'intervention_active': self.trust_interface.intervention_active
        }
        
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
        
        # Reset stuck detection
        self.low_speed_counter = 0
        
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
        self.current_intervention_prob = 0.0
        self.last_step_time = None
        self.trust_interface.reset()
        
        # Reset termination manager
        self.termination_manager.reset()
            
        # Destroy existing vehicle if any
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
        
        # Spawn new vehicle
        self.vehicle, self.spawn_point = spawn_ego_vehicle(self.world)
        
        # Setup sensor manager with the new vehicle
        if self.sensor_manager:
            self.sensor_manager.reset()
            self.sensor_manager.set_vehicle(self.vehicle)
        else:
            self.sensor_manager = SensorManager(self.world, self.vehicle, self.render_mode)
        
        # Generate random waypoints for the new vehicle position
        self.waypoints, self.current_waypoint_idx = generate_random_waypoints(self.vehicle, self.world)
        
        # Setup active scenario if exists
        if self.active_scenario and not self.active_scenario.is_setup:
            self.active_scenario.setup()
        
        # Tick the world to update
        self.world.tick()
        
        # Get initial observation
        obs = get_obs(self.vehicle, self.waypoints, self.current_waypoint_idx, self.waypoint_threshold, self.trust_interface, self.active_scenario)
        
        # Add radar observation to the observation dictionary
        obs['radar_obs'] = self.sensor_manager.get_radar_observation()
        
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
    
    def _adjust_action_based_on_trust(self, action):
        """Adjust the agent's action based on trust level and driving behavior metrics"""
        # Make a copy of the original action
        adjusted_action = np.array(action, dtype=np.float32)
        
        # Get behavior adjustment from trust interface
        behavior = self.trust_interface.get_behavior_adjustment()
        trust_level = behavior['trust_level']
        stability_factor = behavior['stability_factor']
        smoothness_factor = behavior['smoothness_factor']
        hesitation_factor = behavior['hesitation_factor']
        
        # Base intervention probability - higher when trust is lower
        base_intervention_prob = (1.0 - trust_level) * 0.2  # Range: 0.0 to 0.2

        # Higher probability near decision points
        decision_point_factor = 0.04 if self.is_near_decision_point else 0.0
        
        # Combine probabilities
        intervention_prob = base_intervention_prob + decision_point_factor
        
        # Cap at reasonable maximum
        intervention_prob = min(0.5, intervention_prob)  # Max 50% chance per step
        
        # Store the current intervention probability for debugging/visualization
        self.current_intervention_prob = intervention_prob
        self.trust_interface.current_intervention_prob = intervention_prob
        
        # Decide whether to intervene this step
        if np.random.random() > intervention_prob:
            # No intervention this step - return original action unchanged
            return adjusted_action

        # randomly choose intervention type to apply accprding to the facotrs
        intervention_probabilities= [1 - stability_factor, 1 - smoothness_factor, hesitation_factor]
        # Normalize probabilities
        intervention_probabilities = np.array(intervention_probabilities) / (np.sum(intervention_probabilities) + 1e-6)
        intervention_probabilities[-1] = 1 - np.sum(intervention_probabilities[:-1])
        intervention_type = np.random.choice(
            ['steer', 'throttle_or_brake', 'hesitation'], 
            p=intervention_probabilities
        )
        
        if intervention_type == 'steer':
            # 1. Adjust steering based on stability
            # Low stability -> more conservative steering (reduced magnitude)
            steering_adjustment = 0.5 + 0.5 * stability_factor  # Range: 0.5 to 1.0
            adjusted_action[0] *= steering_adjustment
        
        elif intervention_type == 'throttle_or_brake':
            # 2. Adjust throttle/brake based on trust and smoothness
            # Low trust or smoothness -> more gentle acceleration, stronger braking
            if adjusted_action[1] > 0:  # Throttle
                # Low trust or smoothness -> reduce throttle
                throttle_adjustment = 0.3 + 0.7 * (trust_level * smoothness_factor)  # Range: 0.3 to 1.0
                adjusted_action[1] *= throttle_adjustment
                intervention_type = 'throttle'
            else:  # Brake
                # Low trust -> increase braking force
                brake_adjustment = 1.0 + (1.0 - trust_level) * 0.5  # Range: 1.0 to 1.5
                adjusted_action[1] *= brake_adjustment
                intervention_type = 'brake'
        elif intervention_type == 'hesitation':
            # 3. Add hesitation effect (random small delays or reduced actions) 
            if hesitation_factor > 0.3 and np.random.random() < hesitation_factor * 0.5:
                # Occasionally reduce action magnitude to simulate hesitation
                hesitation_reduction = 1.0 - (hesitation_factor * 0.5)  # Range: 0.85 to 0.5
                adjusted_action *= hesitation_reduction
        
        self.trust_interface.record_intervention(intervention_type)
        return adjusted_action
    
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