# Autonomous Driving Reinforcement Learning Pipeline

This document provides a comprehensive overview of the reinforcement learning training pipeline for autonomous driving using CARLA simulator. It explains the code structure, key components, and how different modules interact.

## Table of Contents

1. [Code Structure Overview](#code-structure-overview)
2. [Environment Implementation](#environment-implementation)
3. [Reinforcement Learning Agents](#reinforcement-learning-agents)
4. [Scenarios](#scenarios)
5. [Trust Interface](#trust-interface)
6. [Observation Space](#observation-space)
7. [Action Space](#action-space)
8. [Reward Function](#reward-function)
9. [Training Process](#training-process)
10. [Evaluation Process](#evaluation-process)

## Code Structure Overview

The codebase is organized into several modules:

```
src/
├── agents/             # RL agent implementations
├── environment/        # CARLA environment wrapper
├── mdp/                # MDP components (observations, rewards)
├── scenarios/          # Predefined driving scenarios
├── trust/              # Trust modeling and interface
└── utils/              # Utility functions
```

### Key Files and Their Purposes

- `src/environment/carla_env.py`: Main environment class implementing the Gymnasium interface
- `src/environment/carla_env_discrete.py`: Discrete action space wrapper for DQN
- `src/agents/drl_agent.py`: DRL agent implementation supporting multiple algorithms
- `src/mdp/observation.py`: Observation space definition and processing
- `src/mdp/rewards.py`: Reward function implementation
- `src/trust/trust_interface.py`: Trust modeling and intervention logic
- `src/utils/env_utils.py`: Environment utility functions
- `main.py`: Entry point for training and evaluation

## Environment Implementation

### CarlaEnv Class

The `CarlaEnv` class (`src/environment/carla_env.py`) is the core environment implementation that interfaces with CARLA and follows the Gymnasium API.

#### Key Methods:

- `__init__`: Initializes the CARLA environment, connects to the server, and sets up parameters
- `reset`: Resets the environment for a new episode, spawns the vehicle, and initializes sensors
- `step`: Executes an action, advances the simulation, and returns observation, reward, done status, and info
- `render`: Renders the environment state (camera view with overlays)
- `close`: Cleans up resources when the environment is closed

#### Sensors and Data Collection:

- Camera sensor for visual input
- Collision sensor for detecting collisions
- GPS for localization
- IMU for acceleration and orientation

### CarlaEnvDiscrete Class

The `CarlaEnvDiscrete` class (`src/environment/carla_env_discrete.py`) extends `CarlaEnv` to provide a discrete action space for algorithms like DQN.

#### Key Features:

- Maps discrete action indices to continuous control values
- Configurable discretization levels for steering and throttle/brake
- Provides human-readable action meanings

## Reinforcement Learning Agents

### DRLAgent Class

The `DRLAgent` class (`src/agents/drl_agent.py`) provides a unified interface for different RL algorithms.

#### Supported Algorithms:

- **PPO (Proximal Policy Optimization)**: Default algorithm, good for continuous control
- **SAC (Soft Actor-Critic)**: Sample-efficient algorithm for continuous actions
- **DDPG (Deep Deterministic Policy Gradient)**: Off-policy algorithm for continuous control
- **DQN (Deep Q-Network)**: For discrete action spaces

#### Key Methods:

- `_create_model`: Creates the appropriate RL model based on the selected algorithm
- `train`: Trains the agent on a specific scenario
- `evaluate`: Evaluates the agent's performance on a scenario
- `load`: Loads a trained model from disk

## Scenarios

Scenarios define specific driving tasks and conditions for training and evaluation.

### Base Scenario Classes:

- `ObstacleAvoidanceScenario`: Training for avoiding static obstacles
- `EmergencyBrakingScenario`: Training for emergency braking in response to sudden obstacles
- `LaneSwitchingScenario`: Training for lane changing maneuvers
- `UrbanTrafficScenario`: Complex urban driving with traffic

### Scenario Implementation:

Each scenario implements:
- `setup`: Initializes the scenario (spawns obstacles, sets waypoints)
- `check_scenario_completion`: Determines if the scenario is successfully completed
- `cleanup`: Removes scenario-specific objects and resets the environment

## Trust Interface

The trust interface (`src/trust/trust_interface.py`) models human trust in the autonomous system and provides intervention capabilities.

### Key Components:

- `TrustInterface` class: Manages trust level and interventions
- Trust dynamics model: Updates trust based on system performance
- Intervention mechanism: Allows overriding agent actions when trust is low

### Trust Model:

- Trust level ranges from 0.0 (no trust) to 1.0 (complete trust)
- Trust increases with successful driving and decreases with mistakes
- Trust affects target speed (lower trust = lower allowed speed)
- Trust level is visualized in the environment rendering

### Intervention System:

- Interventions occur when trust falls below a threshold
- During intervention, the system may override agent actions
- Interventions incur a penalty in the reward function

## Observation Space

The observation space (`src/mdp/observation.py`) defines what information is available to the agent.

### Observation Components:

1. **Vehicle State**:
   - Position (x, y coordinates)
   - Velocity (speed and direction)
   - Acceleration
   - Orientation (heading)

2. **Path Information**:
   - Distance to next waypoint
   - Angle to next waypoint
   - Curvature of upcoming path

3. **Environment Information**:
   - Distance to nearest obstacle
   - Road boundaries
   - Traffic light states (if applicable)

4. **Trust Information**:
   - Current trust level
   - Intervention status

### Observation Processing:

- Raw sensor data is processed into a normalized observation vector
- Observations are scaled to appropriate ranges for the neural network
- The observation space is defined as a Dict space with multiple components

## Action Space

### Continuous Action Space (Default):

The default action space is continuous with two dimensions:
1. **Steering**: Range [-1.0, 1.0] where:
   - -1.0: Full left turn
   - 0.0: Straight
   - 1.0: Full right turn

2. **Throttle/Brake**: Range [-1.0, 1.0] where:
   - -1.0: Full brake
   - 0.0: No throttle/brake
   - 1.0: Full throttle

### Discrete Action Space (for DQN):

For DQN, the action space is discretized:
- Configurable number of steering levels (default: 5)
- Configurable number of throttle/brake levels (default: 3)
- Total actions: steering_levels × throttle_brake_levels (default: 15)

### Action Processing:

The `generate_control_from_action` function in `env_utils.py` converts normalized actions to CARLA control commands, handling the conversion between continuous/discrete actions and actual vehicle controls.

## Reward Function

The reward function (`src/mdp/rewards.py`) calculates the reward signal for reinforcement learning.

### Reward Components:

1. **Path Reward** (weight: 0.8):
   - Rewards vehicle for moving in the direction of the path
   - Uses the dot product between velocity vector and path direction
   - Higher when velocity aligns with the path direction
   - Additional bonus for reaching waypoints

2. **Progress Reward** (weight: 0.4):
   - Rewards maintaining the target speed
   - Target speed is adjusted based on trust level
   - Penalizes both too slow and too fast driving

3. **Safety Reward** (weight: 0.2):
   - Penalizes close proximity to other vehicles or obstacles
   - Scales with distance (closer = larger penalty)

4. **Comfort Reward** (weight: 0.02):
   - Penalizes high acceleration and jerk
   - Encourages smooth driving

5. **Trust Reward** (weight: 0.1):
   - Directly incorporates trust level as a reward component
   - Higher trust = higher reward

6. **Intervention Penalty**:
   - Fixed penalty (-1.0) when human intervention occurs
   - Discourages behaviors that lead to interventions

### Reward Calculation:

The `calculate_reward` function combines these components with their respective weights to produce the final reward. The `calculate_path_reward` utility function specifically handles the path alignment reward calculation.

## Training Process

The training process is orchestrated by the `DRLAgent.train()` method.

### Training Steps:

1. **Environment Setup**:
   - Initialize CARLA environment
   - Set up the selected scenario
   - Configure trust interface

2. **Agent Initialization**:
   - Create the appropriate RL algorithm (PPO, SAC, DDPG, or DQN)
   - Set up neural network architecture and hyperparameters

3. **Training Loop**:
   - Agent collects experiences by interacting with the environment
   - Experiences are stored in a buffer (replay buffer for off-policy algorithms)
   - Neural network is updated based on collected experiences
   - Process continues for the specified number of timesteps

4. **Model Saving**:
   - Trained model is saved to disk
   - Naming convention: `{algorithm}_{scenario_name}.zip`

5. **Cleanup**:
   - Resources are released
   - CARLA actors are destroyed

### Hyperparameters:

Key hyperparameters include:
- Learning rate
- Batch size
- Network architecture
- Discount factor (gamma)
- Update frequency

## Evaluation Process

The evaluation process is handled by the `DRLAgent.evaluate()` method.

### Evaluation Steps:

1. **Environment and Scenario Setup**:
   - Similar to training setup

2. **Episode Execution**:
   - Run multiple episodes with the trained policy
   - Use deterministic action selection (no exploration)

3. **Metrics Collection**:
   - Total reward
   - Trust levels
   - Scenario completion rate

4. **Results Reporting**:
   - Average reward
   - Average trust level
   - Completion percentage

### Visualization:

During evaluation, the environment can be rendered to show:
- Camera view from the vehicle
- Trust level visualization
- Reward components
- Waypoints on the path
- Vehicle speed and other metrics

This visualization helps understand the agent's behavior and decision-making process.

## Conclusion

This training pipeline integrates reinforcement learning with trust modeling for autonomous driving. The modular design allows for easy experimentation with different algorithms, scenarios, and reward functions. The trust interface adds a unique dimension by modeling human trust and allowing for interventions, making the system more suitable for human-in-the-loop applications. 