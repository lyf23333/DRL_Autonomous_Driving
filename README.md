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

### Training and Evaluation

You can train and evaluate DRL agents on different scenarios using the main script:

**Training**:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --train --render
```

**Evaluation**:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --eval --render
```

### Model Checkpointing and Loading

The system now supports automatic model checkpointing during training and loading of saved models:

**Training with Custom Checkpoint Frequency**:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --train --checkpoint-freq 50000
```
This will save model checkpoints every 50,000 timesteps in the `models/ppo/checkpoints/` directory.

**Loading a Saved Model**:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --eval --load-model models/ppo/ppo_UrbanTrafficScenario.zip
```

**Resuming Training from a Checkpoint**:
```bash
python3 main.py --scenario urban_traffic --algorithm ppo --train --load-model models/ppo/checkpoints/ppo_UrbanTrafficScenario_1650123456/ppo_UrbanTrafficScenario_steps_100000.zip --resume-training
```

Additional training and model options:
```
--timesteps N          Total number of timesteps for training (default: 100000)
--load-model PATH      Path to a trained model or checkpoint to load
--checkpoint-freq N    Save checkpoint every N timesteps (default: 100000)
--resume-training      Resume training from the loaded model
```

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

# CARLA Track Recording and Visualization

This project extends the CARLA autonomous driving environment with the ability to record vehicle tracks during policy execution and visualize them afterward. The visualization displays the vehicle's path, waypoints, and other vehicles on a top-down view of the environment.

## Features

- **Track Recording**: Record the ego vehicle's position, orientation, and speed during simulation
- **Other Vehicle Tracking**: Optionally track positions of other vehicles in the scene
- **Top-down View**: Capture a top-down camera view of the environment with 1-second interval timelapses
- **Static Visualization**: Generate static visualizations of the vehicle's path with speed color coding
- **Animated Replay**: Create animated visualizations showing the vehicle's movement over time
- **Timelapse Video**: Generate videos from captured top-down image sequences
- **Session Management**: Easily manage and review multiple recording sessions

## Requirements

- Python 3.6 or higher
- OpenCV
- Matplotlib
- NumPy
- CARLA Simulator

## Recording Tracks

The recording functionality is integrated into the `CarlaEnv` class. You can enable recording in your training or evaluation scripts:

```python
# Example usage in a script
from src.environment.carla_env import CarlaEnv

# Create the environment
env = CarlaEnv(config_path="configs/default_config.json")

# Start recording with time-based image capture (1 image per second)
env.start_recording(capture_interval=1.0)

# Run your simulation...
obs = env.reset()
for _ in range(1000):
    action = your_policy(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

# Stop recording and save data (this is also called automatically when env.close() is called)
env.stop_recording()
```

Recording data will be saved to the `recordings/session_YYYY-MM-DD_HH-MM-SS/` directory.

## Top-down Image Capture

You can capture a sequence of top-down images at regular time intervals. This allows you to create timelapse videos of the simulation. To do this:

1. Use the `capture_top_view.py` script which provides a simple way to capture top-down images:

```bash
# Capture top-down images for 30 seconds with 1-second intervals
python capture_top_view.py --duration 30 --capture-interval 1.0

# Specify a different town
python capture_top_view.py --town Town05 --duration 60
```

2. The images are saved in the `recordings/session_YYYY-MM-DD_HH-MM-SS/timelapse/` directory

3. You can then create a video from these images using the `create_timelapse_video.py` script:

```bash
# List all sessions with timelapse images
python create_timelapse_video.py --list

# Create a video from the most recent session
python create_timelapse_video.py

# Create a video from a specific session
python create_timelapse_video.py --session 2

# Customize the frame rate
python create_timelapse_video.py --fps 60

# Specify a custom output path
python create_timelapse_video.py --output my_timelapse.mp4
```

## Visualizing Tracks

Use the `visualize_tracks.py` script to view recorded tracks:

```bash
# List all available recording sessions
python visualize_tracks.py --list

# Visualize the most recent recording session
python visualize_tracks.py

# Visualize a specific recording session
python visualize_tracks.py --recording-dir recordings/session_2023-06-15_14-30-45

# Create an animated visualization
python visualize_tracks.py --animate

# Save visualization to a specific location
python visualize_tracks.py --output my_visualization.png

# Create an animation with a specific framerate
python visualize_tracks.py --animate --fps 60 --output my_animation.mp4

# Hide waypoints or other vehicles
python visualize_tracks.py --no-waypoints --no-other-vehicles
```

## Visualization Features

The visualization includes several useful features:

- **Color-coded track**: The vehicle's path is color-coded by speed
- **Waypoints**: Shows waypoints used for navigation (if available)
- **Other vehicles**: Shows positions of other vehicles in the scene (if recorded)
- **Metadata**: Displays information about the recording session
- **Animation**: The animated visualization shows the vehicle's movement with a trailing path

## Recording Data Structure

Each recording session creates a directory with the following files:

- `top_down_view.png`: A top-down image of the environment
- `track_data.json`: The ego vehicle's position, orientation, and speed data
- `waypoints_data.json`: The waypoints used for navigation
- `other_vehicles_data.json`: Data about other vehicles in the scene (if recorded)
- `metadata.json`: Information about the recording session
- `timelapse/`: Directory containing timestamped top-down images (if using time-based capture)
- `timelapse_video.mp4`: Video generated from the timelapse images (if created)

## Example Visualizations

Static visualization:
![Example Track Visualization](recordings/session_example/track_visualization.png)

Animation (click to see):
[![Track Animation](recordings/session_example/track_visualization.png)](recordings/session_example/track_animation.mp4)

## Integration with Your Project

The recording functionality is built into the `CarlaEnv` class, so it can be easily integrated with existing training or evaluation code. Simply call `env.start_recording()` before starting your simulation and `env.stop_recording()` when finished.

## Extending the Visualization

The visualization script is designed to be extensible. You can modify the `plot_static_tracks` and `create_animated_visualization` functions in `visualize_tracks.py` to add additional features or customize the visualization to your needs.
