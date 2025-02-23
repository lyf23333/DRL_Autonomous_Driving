# DRL Autonomous Driving with Trust-Based Adaptation

This project explores how Deep Reinforcement Learning (DRL) can personalize autonomous vehicle behavior based on dispositional trust levels in CARLA simulation environment.

## Project Overview
The project focuses on adapting autonomous vehicle behavior based on user's dispositional trust - the baseline level of trust a person has in automated systems. The system uses deep reinforcement learning to dynamically adjust driving behavior according to trust feedback. This project focuses on evaluating the effect of different deep reinforcement learning algorithms on the performance of the system.

## Prerequisites
- Ubuntu 20.04
- Python 3.7
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
3. **Verify CARLA Installation**
```bash
# Start CARLA simulator
cd /opt/carla-simulator
./CarlaUE4.sh -quality-level=Low
```

3. **Set up Python Environment**
```bash
# Create and activate virtual environment
conda create -n carla python=3.9
conda activate carla

# Install dependencies
cd /opt/carla-simulator/PythonAPI/examples
pip3 install -r requirements.txt
```

4. **Set up Python Environment for this repo**
mkdir -p ~/git && cd ~/git
git clone git@github.com:lyf23333/DRL_Autonomous_Driving.git
pip3 install -r requirements.txt


## Project Structure
```
.
├── scenarios/           # Predefined driving scenarios
├── src/
│   ├── environment/    # CARLA environment wrapper
│   ├── agents/        # DRL agents implementation
│   ├── trust/         # Trust feedback and adaptation mechanisms
│   └── utils/         # Utility functions
├── config/            # Configuration files
├── data/             # Data collection and storage
├── scripts/          # Utility scripts
└── tests/            # Test cases
```

## Running Scenarios

### Automated Training/Evaluation
```bash
python src/main.py --scenario [scenario_name] --algorithm [algorithm_name] --train
python src/main.py --scenario [scenario_name] --algorithm [algorithm_name] --eval
```

### Manual Scenario Testing
The project includes a manual testing script that allows you to control the vehicle and test different scenarios:

1. Start CARLA simulator:
```bash
cd /opt/carla-simulator
./CarlaUE4.sh -quality-level=Low
```

2. Run the manual testing script, in which you can control the vehicle by keyboard:
```bash
python3 scripts/manual_scenario_test.py --scenario [scenario_name]
```

3. Run the automatic testing script, in which the vehicle is controlled by a PID controller:
```bash
python3 scripts/automatic_scenario_test.py --scenario [scenario_name]
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
The system collects trust-related feedback through:
- Manual intervention monitoring
- Real-time trust level inputs
- Performance metrics tracking

## Evaluation Scenarios
1. Lane Switching
2. Urban Traffic Navigation
3. Obstacle Avoidance
4. Emergency Braking
5. Pedestrian Interaction (TODO)


## License
This project is licensed under the MIT License - see the LICENSE file for details.
