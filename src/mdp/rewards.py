import numpy as np
from src.utils.env_utils import calculate_path_reward


def calculate_reward(vehicle, waypoints, current_waypoint_idx, waypoint_threshold, trust_interface, active_scenario, world, target_speed):
    """Calculate reward based on current state"""
    if vehicle is None:
        return 0.0
        
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
    if active_scenario:
        danger_threshold = 5.0  # meters
        min_distance = float('inf')
        ego_location = vehicle.get_location()
        
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.id != vehicle.id:
                distance = ego_location.distance(vehicle.get_location())
                min_distance = min(min_distance, distance)
        
        if min_distance < danger_threshold:
            safety_reward = -1.0 * (1.0 - min_distance / danger_threshold)
    
    # Trust-based reward
    trust_reward = trust_interface.trust_level if trust_interface else 0.5
    
    # Intervention penalty
    intervention_penalty = -1.0 if (trust_interface and trust_interface.intervention_active) else 0.0
    
    # Combine rewards with weights
    path_reward_weighted = 0.4 * path_reward
    progress_reward_weighted = 0.4 * progress_reward
    safety_reward_weighted = 0.2 * safety_reward
    trust_reward_weighted = 0.1 * trust_reward
    
    # Store reward components for visualization
    reward_components = {
        'path': path_reward_weighted,
        'progress': progress_reward_weighted,
        'safety': safety_reward_weighted,
        'trust': trust_reward_weighted,
        'intervention': intervention_penalty
    }
    
    total_reward = (
        path_reward_weighted +
        progress_reward_weighted +
        safety_reward_weighted +
        trust_reward_weighted +
        intervention_penalty
    )
    
    return total_reward, reward_components
