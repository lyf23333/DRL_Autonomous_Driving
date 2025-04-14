"""
Configuration for the CARLA environment.
This module contains default configuration settings for the CarlaEnv class.
"""

from dataclasses import dataclass
import os
import json
from typing import Dict, List, Optional, Any, Union

@dataclass
class CarlaEnvConfig:
    """Configuration for the CARLA environment"""
    
    # Connection settings
    town: str = 'Town01'
    port: int = 2000
    host: str = 'localhost'
    timeout: float = 10.0
    
    # Episode settings
    max_episode_steps: int = 1000
    
    # Visualization settings
    render_mode: Optional[str] = None
    camera_width: int = 800
    camera_height: int = 600
    trust_viz_height: int = 280
    show_waypoints: bool = True
    waypoint_lookahead: int = 20
    
    # History tracking
    max_trust_history: int = 100
    max_reward_history: int = 100
    location_history_length: int = 5
    action_history_length: int = 5  # Number of past actions to include in observation
    
    # Path following settings
    waypoint_threshold: float = 2.0  # meters
    path_length: int = 20  # number of waypoints
    num_observed_waypoints: int = 3  # number of waypoints to include in observation
    
    # Decision point detection
    decision_point_distance: float = 20.0  # meters
    
    # Speed settings
    base_target_speed: float = 20.0  # km/h at max trust
    min_target_speed: float = 5.0    # km/h at min trust
    
    # Trust settings
    initial_trust: float = 0.75
    trust_change_rate: float = 0.05
    
    # Action settings
    max_steering_angle: float = 1.0
    max_throttle: float = 1.0
    max_brake: float = 1.0
    steering_smoothing: float = 0.2
    throttle_smoothing: float = 0.1
    
    # Observation settings
    radar_range: float = 20.0  # meters
    radar_resolution: float = 3.0  # degrees per radar observation point
    max_detectable_vehicles: int = 3
    
    # Reward component weights
    reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default values for complex types"""
        if self.reward_weights is None:
            self.reward_weights = {
                'path': 1.0,       # Following the prescribed path
                'progress': 1.0,   # Making progress along the path
                'safety': 1.0,     # Avoiding collisions and staying on road
                'comfort': 0.5,    # Smooth driving (acceleration, jerk)
                'trust': 0.5,      # Maintaining high trust level
                'intervention': -1.0  # Penalty for interventions
            }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CarlaEnvConfig':
        """Create a config object from a dictionary"""
        # Filter out keys that aren't in the dataclass
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'CarlaEnvConfig':
        """Load configuration from a JSON file"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            # Convert None, complex objects, etc. to JSON-serializable format
            if hasattr(value, 'to_dict'):
                result[field] = value.to_dict()
            else:
                result[field] = value
        return result
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def update(self, **kwargs) -> 'CarlaEnvConfig':
        """Update config with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"CarlaEnvConfig has no attribute '{key}'")
        return self 