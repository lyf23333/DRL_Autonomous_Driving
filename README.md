# DRL Autonomous Driving with Trust-Based Adaptation

This project explores how Deep Reinforcement Learning (DRL) can personalize autonomous vehicle behavior based on dispositional trust levels in CARLA simulation environment.

## Project Overview
The project focuses on adapting autonomous vehicle behavior based on user's dispositional trust - the baseline level of trust a person has in automated systems. The system uses deep reinforcement learning to dynamically adjust driving behavior according to trust feedback. This project focuses on evaluating the effect of different deep reinforcement learning algorithms on the performance of the system.

## Prerequisites
- Ubuntu 20.04
- Python 3.8
- CARLA Simulator 0.9.15
- GPU with at least 8GB memory (recommended)

**Note**: Have not tested in windows yet

## Installation Steps

1. **Install CARLA Simulator**
Follow the guiline described [here](https://carla.readthedocs.io/en/latest/start_quickstart/) to install carla simulator and python api required pacakge.
Or you can follow the guide here:

```bash
# Download CARLA 0.9.15
wget https://github.com/carla-simulator/carla/archive/refs/tags/0.9.15.tar.gz

# Extract to your preferred location (e.g., /opt/carla-simulator)
sudo mkdir -p /opt/carla-simulator
sudo tar -xf CARLA_0.9.15.tar.gz -C /opt/carla-simulator

# Add CARLA Python API to PYTHONPATH
echo 'export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg' >> ~/.bashrc
source ~/.bashrc
```

2. **Verify CARLA Installation**
```bash
# Start CARLA simulator
cd /opt/carla-simulator
./CarlaUE4.sh -quality-level=Low
```

3. **Set up Python Environment**
If you don't have miniconda installed, refer to [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
```bash
# Create and activate virtual environment
conda create -n carla python=3.8
conda activate carla

# Install dependencies
cd /opt/carla-simulator/PythonAPI/examples
pip3 install -r requirements.txt
```

4. **Set up Python Environment for this repo**
```bash
mkdir -p ~/git && cd ~/git
git clone git@github.com:lyf23333/DRL_Autonomous_Driving.git
pip3 install -r requirements.txt
```

## Project Structure
```
.
├── config/            # Configuration files
├── scenarios/         # Predefined driving scenarios
├── scripts/           # Utility scripts
├── src/
│   ├── environment/    # CARLA environment wrapper
│   ├── agents/        # DRL agents implementation
│   ├── trust/         # Trust feedback and adaptation mechanisms
│   └── utils/         # Utility functions
├── data/             # Data collection and storage
└── tests/            # Test cases
```

## Running Scenarios

### Integrated CARLA Server Management

The project now includes an integrated CARLA server manager that can automatically start and stop the CARLA simulator as part of your training or evaluation pipeline. This eliminates the need to manually start CARLA in a separate terminal.

To use this feature, add the `--start-carla` flag to your command:

**Training**:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --train --render --start-carla
```
**Evaluation**:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --eval --render --start-carla
```

Additional CARLA server configuration options:

```
--carla-path PATH      Path to CARLA installation (if not set, will try to auto-detect)
--port PORT            Port to run CARLA server on (default: 2000)
--town TOWN            CARLA town/map to use (default: Town01)
--quality {Low,Epic}   Graphics quality for CARLA (default: Epic)
--offscreen            Run CARLA in offscreen mode (no rendering)
```

Examples:

1. Start CARLA with a specific town and port:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --train --start-carla --town Town05 --port 2050
```

2. Run in low-quality mode for better performance:
```bash
python3 main.py --scenario lane_switching --algorithm sac --train --start-carla --quality Low
```

3. Run in headless mode (useful for servers without a display):
```bash
python3 main.py --scenario obstacle_avoidance --algorithm ppo --train --start-carla --offscreen
```

4. Specify a custom CARLA installation path:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --train --start-carla --carla-path /path/to/carla
```

The CARLA server will automatically shut down when your script exits, ensuring clean termination of all processes.

### Manual Scenario Testing
The project includes a manual testing script that allows you to control the vehicle and test different scenarios:

1. You can now start CARLA automatically with the test scripts:
```bash
python3 scripts/manual_scenario_test.py --scenario obstacle_avoidance --start-carla
```

Or manually start CARLA simulator:
```bash
cd /opt/carla-simulator
./CarlaUE4.sh -quality-level=Low
```

2. Run the manual testing script, in which you can control the vehicle by keyboard.
In this case, you are in the role of the **RL agent/driver**, and the trust interface is automatically updated based on your driving behavior.
```bash
python3 scripts/manual_scenario_test.py --scenario [scenario_name]
```

3. Run the automatic testing script, in which the vehicle is controlled by a PID controller. In this case, you are in the role of the **human on the autonomous vehicle** and can provide real-time trust feedback to the RL agent by pressing space to brake. The agent is a PID controller that is set to adapt the speed based on the trust level.
```bash
python3 scripts/automatic_scenario_test.py --scenario [scenario_name]
```

Both test scripts support the same CARLA server configuration options as the main script:
```
--start-carla           Start CARLA server automatically
--carla-path PATH       Path to CARLA installation
--port PORT             Port to run CARLA server on (default: 2000)
--town TOWN             CARLA town/map to use (default: Town01)
--quality {Low,Epic}    Graphics quality for CARLA (default: Epic)
--offscreen             Run CARLA in offscreen mode (no rendering)
```

Available scenarios:
- `lane_switching`: Test lane changing behavior
- `urban_traffic`: Navigate through urban traffic with vehicles and pedestrians
- `obstacle_avoidance`: Avoid static and dynamic obstacles
- `emergency_braking`: Test reaction to sudden braking events and emergency stops

Controls:
- Arrow Keys: Control the vehicle
  - ↑: Accelerate
  - ↓: Brake
  - ←/→: Steer
- Space: Record manual intervention
- R: Reset episode
- ESC: Quit

The manual testing interface provides real-time information about:
- Vehicle controls (steering, throttle, brake)
- Trust level
- Accumulated reward
- Episode statistics
- Scenario completion status

## Trust Feedback Mechanisms
The system employs a multi-faceted approach to collect and adapt to trust-related feedback, which is crucial for personalizing the autonomous driving experience. The trust feedback mechanisms include:

1. **Manual Intervention Monitoring**: 
   - The system tracks instances where the user manually intervenes in the vehicle's operation. Frequent interventions may indicate a lower level of trust in the system's autonomous capabilities.
   - These interventions are logged and analyzed to adjust the vehicle's behavior dynamically, aiming to reduce the need for future interventions.

2. **Real-Time Trust Level Inputs**:
   - Users can provide real-time feedback on their trust level through a user interface. This feedback is used to adjust the driving style, such as being more cautious or aggressive, depending on the user's comfort level.
   - The system uses this input to fine-tune the reinforcement learning model, ensuring that the vehicle's behavior aligns with the user's trust preferences.

3. **Performance Metrics Tracking**:
   - The system continuously monitors various performance metrics, such as the vehicle's adherence to traffic rules, smoothness of driving, and response to dynamic obstacles.
   - These metrics are used to infer the user's trust level indirectly. For example, consistent adherence to traffic rules may enhance trust, while erratic driving may diminish it.
   - The feedback from these metrics is integrated into the learning algorithm to improve the vehicle's decision-making process.

4. **Modeling Trust Feedback**:
   - Trust feedback is modeled by associating specific driving behaviors with trust levels. For instance, frequent or abrupt braking is interpreted as a sign of low trust, prompting the system to adapt by driving more cautiously.
   - The adaptation process involves smoothing out the driving style to gradually rebuild trust. This includes reducing sudden accelerations or decelerations and maintaining a safe distance from other vehicles.
   - The reinforcement learning model is updated with these trust indicators, allowing the vehicle to learn and adjust its behavior in real-time to better match the user's trust level.

## Trust-Based Behavior Adaptation

The system implements a comprehensive trust-based behavior adaptation mechanism that modifies the vehicle's driving style based on the current trust level and driving metrics. This adaptation affects multiple aspects of driving behavior:

### 1. Speed Adjustment

- **Target Speed**: The vehicle's target speed is directly proportional to the trust level. Higher trust leads to higher target speeds, while lower trust results in more conservative speeds.
- **Acceleration Profile**: At low trust levels, the vehicle accelerates more gently to avoid sudden movements that might decrease trust further.
- **Braking Intensity**: When trust is low, the vehicle applies stronger braking earlier to maintain a larger safety margin.

### 2. Steering Behavior

- **Steering Magnitude**: Lower trust levels result in more conservative steering inputs, reducing the maximum steering angle to avoid sharp turns.
- **Steering Stability**: The system monitors steering stability and adjusts control parameters to maintain smoother trajectories when stability is low.
- **Turn Anticipation**: At low trust levels, the vehicle begins turns earlier and executes them more gradually.

### 3. Hesitation Modeling

- **Decision Points**: The system detects decision points such as intersections and lane merges, where hesitation is more likely to occur.
- **Hesitation Effects**: When trust is low, the vehicle occasionally exhibits hesitation behaviors, such as temporarily reducing action magnitudes or introducing small delays.
- **Confidence Building**: As trust increases, hesitation behaviors are gradually eliminated, resulting in more decisive driving.

### 4. Smoothness and Comfort

- **Action Smoothing**: Lower trust levels lead to more gradual changes in control inputs, avoiding abrupt transitions.
- **Jerk Minimization**: The system reduces acceleration and deceleration rates to minimize jerk (rate of change of acceleration) when trust is low.
- **Predictable Behavior**: At low trust levels, the vehicle maintains more consistent and predictable behavior patterns.

### Implementation Details

The trust-based behavior adaptation is implemented through several key components:

1. **Behavior Adjustment Factors**: The system calculates and maintains behavior adjustment factors based on trust level and driving metrics:
   - `trust_level`: The current trust level (0.0 to 1.0)
   - `behavior_factor`: Combined behavior factor incorporating stability, smoothness, and hesitation
   - `stability_factor`: Measure of steering stability
   - `smoothness_factor`: Measure of acceleration and braking smoothness
   - `hesitation_factor`: Measure of confidence vs. hesitation

2. **Action Adjustment**: The `_adjust_action_based_on_trust` method modifies the agent's actions based on these factors:
   - Steering adjustments based on stability
   - Throttle/brake adjustments based on trust and smoothness
   - Random hesitation effects when trust is low

3. **PID Control Parameters**: For the automatic controller, PID parameters are dynamically adjusted based on trust:
   - Lower gains at low trust levels for more conservative control
   - Smaller maximum steering changes for smoother transitions
   - Adjusted maximum throttle and braking forces

4. **Visualization**: The system provides visual feedback on trust-based behavior adjustments through the user interface, displaying:
   - Current trust level and behavior factors
   - Driving metrics and their impact on behavior
   - Intervention probability and history

This comprehensive trust-based behavior adaptation system creates a more natural and human-like driving experience that responds appropriately to the user's trust level, helping to build and maintain trust over time.

## Evaluation Scenarios
1. Lane Switching
2. Urban Traffic Navigation
3. Obstacle Avoidance
4. Emergency Braking
5. Pedestrian Interaction (TODO)


## License
This project is licensed under the MIT License - see the LICENSE file for details.
