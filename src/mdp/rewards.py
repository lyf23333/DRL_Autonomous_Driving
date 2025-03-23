import numpy as np
from src.utils.env_utils import calculate_path_reward


def calculate_reward(vehicle, waypoints, current_waypoint_idx, waypoint_threshold, trust_interface=None, scenario=None, world=None, target_speed=None, reward_weights=None):
    """
    Calculate the reward for the current state

    Args:
        vehicle: The ego vehicle
        waypoints: List of waypoints to follow
        current_waypoint_idx: Index of the current waypoint
        waypoint_threshold: Distance threshold to consider a waypoint reached
        trust_interface: Trust interface object for trust-based rewards
        scenario: Active scenario object
        world: CARLA world object
        target_speed: Target speed for the vehicle
        reward_weights: Dictionary of weights for each reward component

    Returns:
        total_reward: The total reward
        reward_components: Dictionary of individual reward components
    """
    if vehicle is None or len(waypoints) == 0:
        return 0.0, {}
        
    # Default reward weights
    if reward_weights is None:
        reward_weights = {
            'path': 1.0,       # Following the prescribed path
            'progress': 1.0,   # Making progress along the path
            'safety': 1.0,     # Avoiding collisions and staying on road
            'comfort': 0.5,    # Smooth driving (acceleration, jerk)
            'trust': 0.5,      # Maintaining high trust level
            'intervention': -1.0  # Penalty for interventions
        }
        
    # Initialize reward components
    reward_components = {
        'path': 0.0,
        'progress': 0.0,
        'safety': 0.0,
        'comfort': 0.0,
        'trust': 0.0,
        'intervention': 0.0
    }
    
    # Get current vehicle state
    velocity = vehicle.get_velocity()
    current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # km/h
    
    # Path following reward - using the new utility function
    path_reward = calculate_path_reward(vehicle, waypoints, current_waypoint_idx, waypoint_threshold, target_speed)
    
    # Progress reward (based on speed)
    # Use trust-based target speed instead of fixed value
    speed_diff = abs(current_speed - target_speed)
    progress_reward = 1.0 - min(1.0, speed_diff / max(1.0, target_speed))  # Avoid division by zero
    
    # Safety reward components
    safety_reward = 0.0
    
    # Trust-based reward
    trust_reward = 0.0
    
    # Intervention penalty
    intervention_penalty = 0.0
    
    # Store reward components for visualization
    reward_components['path'] = path_reward
    reward_components['progress'] = progress_reward
    reward_components['safety'] = safety_reward
    reward_components['trust'] = trust_reward
    reward_components['intervention'] = intervention_penalty
    
    # Weight and sum the reward components
    total_reward = 0.0
    for component, value in reward_components.items():
        if component in reward_weights:
            reward_components[component] = value * reward_weights[component]
            total_reward += reward_components[component]
            
    return total_reward, reward_components
