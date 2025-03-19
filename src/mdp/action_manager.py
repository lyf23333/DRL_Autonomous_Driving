"""
Action Manager for the CARLA environment.
This module defines the action space and handles action processing.
"""

import numpy as np
from gymnasium import spaces
import carla

class ActionManager:
    """
    Manages the action space and processing for the CARLA environment.
    """
    
    def __init__(self, config):
        """
        Initialize the action manager.
        
        Args:
            config: Configuration object with action settings
        """
        self.config = config
        
        # Define the action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # [steering, throttle/brake]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action limits
        self.max_steering_angle = getattr(config, 'max_steering_angle', 1.0)
        self.max_throttle = getattr(config, 'max_throttle', 1.0)
        self.max_brake = getattr(config, 'max_brake', 1.0)
        
        # Action smoothing parameters
        self.steering_smoothing = getattr(config, 'steering_smoothing', 0.2)
        self.throttle_smoothing = getattr(config, 'throttle_smoothing', 0.1)
        
        # Previous action for smoothing
        self.previous_action = None
    
    def process_action(self, action):
        """
        Process the action from the agent.
        
        Args:
            action: Raw action from the agent [steering, throttle/brake]
            
        Returns:
            processed_action: Processed action after normalization and constraints
        """
        # Make a copy to avoid modifying the original
        processed_action = np.array(action, dtype=np.float32)
        
        # Ensure action is within bounds
        processed_action = np.clip(processed_action, self.action_space.low, self.action_space.high)
        
        # Apply smoothing if there's a previous action
        if self.previous_action is not None:
            processed_action[0] = (1 - self.steering_smoothing) * self.previous_action[0] + self.steering_smoothing * processed_action[0]
            processed_action[1] = (1 - self.throttle_smoothing) * self.previous_action[1] + self.throttle_smoothing * processed_action[1]
        
        # Store current action for next step
        self.previous_action = processed_action.copy()
        
        return processed_action
    
    def generate_vehicle_control(self, action):
        """
        Convert normalized action to CARLA vehicle control.
        
        Args:
            action: Normalized action [steering, throttle/brake]
            
        Returns:
            control: CARLA VehicleControl object
        """
        # Ensure action is processed
        processed_action = self.process_action(action)
        
        # Create CARLA vehicle control
        control = carla.VehicleControl()
        
        # Set steering [-1, 1] -> [-max_steering, max_steering]
        control.steer = float(processed_action[0] * self.max_steering_angle)
        
        # Split throttle and brake
        if processed_action[1] >= 0:
            # Positive value means throttle
            control.throttle = float(processed_action[1] * self.max_throttle)
            control.brake = 0.0
        else:
            # Negative value means brake
            control.throttle = 0.0
            control.brake = float(-processed_action[1] * self.max_brake)
        
        # Fixed values
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        
        return control
    
    def adjust_action_with_trust(self, action, trust_interface, is_near_decision_point):
        """
        Adjust the agent's action based on trust level and driving behavior metrics.
        
        Args:
            action: Original action from the agent
            trust_interface: Trust interface object
            is_near_decision_point: Whether the vehicle is near a decision point
            
        Returns:
            adjusted_action: Action after trust-based adjustments
        """
        # Make a copy of the original action
        adjusted_action = np.array(action, dtype=np.float32)
        
        # Get behavior adjustment from trust interface
        behavior = trust_interface.get_behavior_adjustment()
        trust_level = behavior['trust_level']
        stability_factor = behavior['stability_factor']
        smoothness_factor = behavior['smoothness_factor']
        hesitation_factor = behavior['hesitation_factor']
        
        # Base intervention probability - higher when trust is lower
        base_intervention_prob = (1.0 - trust_level) * 0.2  # Range: 0.0 to 0.2

        # Higher probability near decision points
        decision_point_factor = 0.04 if is_near_decision_point else 0.0
        
        # Combine probabilities
        intervention_prob = base_intervention_prob + decision_point_factor
        
        # Cap at reasonable maximum
        intervention_prob = min(0.5, intervention_prob)  # Max 50% chance per step
        
        # Store the current intervention probability for debugging/visualization
        current_intervention_prob = intervention_prob
        trust_interface.current_intervention_prob = intervention_prob
        
        # Decide whether to intervene this step
        if np.random.random() > intervention_prob:
            # No intervention this step - return original action unchanged
            return adjusted_action, current_intervention_prob

        # Randomly choose intervention type to apply according to the factors
        intervention_probabilities = [1 - stability_factor, 1 - smoothness_factor, hesitation_factor]
        # Normalize probabilities
        intervention_probabilities = np.array(intervention_probabilities) / (np.sum(intervention_probabilities) + 1e-6)
        intervention_probabilities[-1] = 1 - np.sum(intervention_probabilities[:-1])
        intervention_type = np.random.choice(
            ['steer', 'throttle_or_brake', 'hesitation'], 
            p=intervention_probabilities
        )
        
        if intervention_type == 'steer':
            # 1. Adjust steering based on stability and trust
            # Low stability or trust -> more conservative steering (reduced magnitude)
            steering_adjustment = 0.3 + 0.7 * (trust_level * stability_factor)  # Range: 0.3 to 1.0
            adjusted_action[0] *= steering_adjustment
            
            # For very unstable driving, we may need to reduce steering more aggressively
            if stability_factor < 0.4 and abs(adjusted_action[0]) > 0.5:
                # Apply additional reduction for large steering inputs when stability is low
                additional_reduction = max(0.5, stability_factor)
                adjusted_action[0] *= additional_reduction
        
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
            if hesitation_factor > 0.3:
                # Reduce action magnitude to simulate hesitation
                hesitation_reduction = 1.0 - (hesitation_factor * 0.5)  # Range: 0.85 to 0.5
                adjusted_action *= hesitation_reduction
        
        trust_interface.record_intervention(intervention_type)
        return adjusted_action, current_intervention_prob
    
    def reset(self):
        """Reset the action manager state"""
        self.previous_action = None 