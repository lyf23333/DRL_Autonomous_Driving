"""Default configuration for the DRL autonomous driving project"""

# CARLA simulation settings
CARLA_CONFIG = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 10.0,
    'quality_level': 'Low',  # Options: Low, Epic
    'frame_rate': 20,
    'width': 800,
    'height': 600
}

# DRL training settings
TRAINING_CONFIG = {
    'algorithms': {
        'ppo': {
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'clip_range': 0.2
        },
        'sac': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 100,
            'batch_size': 256,
            'train_freq': 1,
            'gradient_steps': 1
        },
        'ddpg': {
            'learning_rate': 1e-3,
            'buffer_size': 100000,
            'learning_starts': 100,
            'batch_size': 128
        }
    },
    'total_timesteps': 100000,
    'eval_episodes': 10,
    'save_freq': 10000
}

# Trust adaptation settings
TRUST_CONFIG = {
    'initial_trust_level': 0.5,
    'trust_decay_rate': 0.1,
    'trust_increase_rate': 0.05,
    'intervention_threshold': 5,
    'min_trust_level': 0.0,
    'max_trust_level': 1.0
}

# Scenario settings
SCENARIO_CONFIG = {
    'lane_switching': {
        'min_distance': 10.0,
        'max_speed': 30.0,
        'lane_change_distance': 5.0
    },
    'urban_traffic': {
        'num_vehicles': 10,
        'num_pedestrians': 5,
        'traffic_light_duration': {
            'green': 5.0,
            'yellow': 2.0,
            'red': 5.0
        }
    },
    'obstacle_avoidance': {
        'num_static_obstacles': 5,
        'num_dynamic_obstacles': 2,
        'min_obstacle_distance': 2.0,
        'sensor_range': 20.0
    }
}

# Reward function weights
REWARD_WEIGHTS = {
    'progress': 1.0,
    'safety': 2.0,
    'comfort': 0.5,
    'trust_level': 1.0,
    'intervention_penalty': -1.0,
    'collision_penalty': -5.0,
    'goal_reward': 10.0
} 