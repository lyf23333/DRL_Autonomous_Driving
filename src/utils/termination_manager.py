import numpy as np

class TerminationManager:
    """
    Manages termination and truncation conditions for the CARLA environment.
    
    In reinforcement learning environments:
    - Termination: Natural end of an episode due to failure or impossible recovery.
      The agent receives no further rewards after termination.
    - Truncation: Artificial end of an episode due to time limits or successful completion.
      The episode could have continued, but we choose to end it.
    """
    
    def __init__(self, max_episode_steps=1000):
        """
        Initialize the termination manager.
        
        Args:
            max_episode_steps: Maximum number of steps before truncation
        """
        self.max_episode_steps = max_episode_steps
        self.low_speed_counter = 0
        
    def check_termination(self, vehicle, world, step_count, waypoints, current_waypoint_idx, 
                          active_scenario, collision_detected):
        """
        Check if the episode should be terminated or truncated.
        
        Termination conditions (failure states):
        - Vehicle doesn't exist
        - Vehicle is off-road
        - Vehicle is stuck (very low speed for extended time)
        - Collision detected (if configured to terminate on collision)
        
        Truncation conditions (success or time limit):
        - Scenario is completed successfully
        - Vehicle reached the end of the path
        - Maximum number of steps reached
        
        Args:
            vehicle: CARLA vehicle object
            world: CARLA world object
            step_count: Current step count
            waypoints: List of waypoints
            current_waypoint_idx: Index of the current waypoint
            active_scenario: Active scenario object
            collision_detected: Whether a collision was detected
            
        Returns:
            terminated: True if the episode should be terminated (failure state)
            truncated: True if the episode should be truncated (success or time limit)
        """
        # Initialize return values
        terminated = False
        truncated = False
        
        # Check if vehicle exists
        if vehicle is None:
            terminated = True
            return terminated, truncated
            
        # Check if active scenario is completed
        if active_scenario and active_scenario.check_scenario_completion():
            truncated = True
            
        # Check if vehicle reached the end of the path - TRUNCATE
        if waypoints and current_waypoint_idx >= len(waypoints):
            print("Episode truncated: Reached end of path")
            truncated = True
            
        # Check if vehicle is off-road - TERMINATE
        current_waypoint = world.get_map().get_waypoint(vehicle.get_location())
        if current_waypoint is None:
            print("Episode truncated: Vehicle is off-road")
            truncated = True
            
        # Check if vehicle is stuck (very low speed for extended time) - TERMINATE
        velocity = vehicle.get_velocity()
        current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # km/h
        
        if current_speed < 1.0:  # Less than 1 km/h
            self.low_speed_counter += 1
        else:
            self.low_speed_counter = 0
            
        if self.low_speed_counter > 100:  # Stuck for too long
            print("Episode truncated: Vehicle is stuck")
            truncated = True
            
        # Check if maximum episode length reached - TRUNCATE
        if step_count >= self.max_episode_steps:
            print(f"Episode truncated: Reached maximum episode length ({self.max_episode_steps} steps)")
            truncated = True

        # Check for collision
        if collision_detected:
            print("Episode terminated: Collision detected")
            terminated = True
            
        return terminated, truncated
        
    def reset(self):
        """Reset the termination manager state"""
        self.low_speed_counter = 0 