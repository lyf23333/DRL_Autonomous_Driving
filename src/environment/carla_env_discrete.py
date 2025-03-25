import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.environment.carla_env import CarlaEnv
import carla

class CarlaEnvDiscrete(CarlaEnv):
    """
    A wrapper around CarlaEnv that provides a discrete action space.
    This makes it compatible with algorithms like DQN that require discrete actions.
    """
    
    def __init__(self, trust_interface, config,
                 steering_levels=5, throttle_brake_levels=3):
        """
        Initialize the discrete action space environment.
        
        Args:
            town: CARLA town to use
            port: CARLA server port
            trust_interface: Trust interface object
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            steering_levels: Number of discrete steering levels
            throttle_brake_levels: Number of discrete throttle/brake levels
        """
        # Initialize the parent class
        super(CarlaEnvDiscrete, self).__init__( trust_interface=trust_interface, config=config)
        
        # Define discrete action space parameters
        self.steering_levels = steering_levels
        self.throttle_brake_levels = throttle_brake_levels
        
        # Calculate total number of actions
        self.num_actions = steering_levels * throttle_brake_levels
        
        # Override the action space with a discrete space
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Create mappings from discrete actions to continuous values
        self._create_action_mappings()
        
    def _create_action_mappings(self):
        """Create mappings from discrete action indices to continuous control values"""
        # Create steering values from -1.0 to 1.0
        self.steering_values = np.linspace(-1.0, 1.0, self.steering_levels)
        
        # Create throttle/brake values from -1.0 (full brake) to 1.0 (full throttle)
        self.throttle_brake_values = np.linspace(-1.0, 1.0, self.throttle_brake_levels)
        
        # Create a lookup table for all action combinations
        self.action_map = []
        for throttle_brake in self.throttle_brake_values:
            for steering in self.steering_values:
                self.action_map.append((steering, throttle_brake))
                
        # Print action space information
        print(f"Discrete action space created with {self.num_actions} actions:")
        for i, (steer, throttle_brake) in enumerate(self.action_map):
            print(f"  Action {i}: Steering = {steer:.2f}, Throttle/Brake = {throttle_brake:.2f}")
    
    def step(self, action):
        """
        Take a step in the environment using a discrete action.
        
        Args:
            action: Integer representing the discrete action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_actions-1}")
        
        # Convert discrete action to continuous values
        continuous_action = self.action_map[action]
        
        # Call the parent class step method with the continuous action
        return super(CarlaEnvDiscrete, self).step(continuous_action)
    
    def get_action_meaning(self, action):
        """Get the meaning of a discrete action as a human-readable string"""
        if not 0 <= action < self.num_actions:
            return "Invalid action"
            
        steering, throttle_brake = self.action_map[action]
        
        # Interpret steering
        if steering < -0.2:
            steer_text = f"Turn Left ({steering:.2f})"
        elif steering > 0.2:
            steer_text = f"Turn Right ({steering:.2f})"
        else:
            steer_text = f"Straight ({steering:.2f})"
            
        # Interpret throttle/brake
        if throttle_brake > 0.2:
            tb_text = f"Accelerate ({throttle_brake:.2f})"
        elif throttle_brake < -0.2:
            tb_text = f"Brake ({throttle_brake:.2f})"
        else:
            tb_text = f"Coast ({throttle_brake:.2f})"
            
        return f"{steer_text}, {tb_text}" 