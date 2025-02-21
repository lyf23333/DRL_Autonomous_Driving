# DRL Autonomous Driving with Trust-Based Adaptation

This project explores how Deep Reinforcement Learning (DRL) can personalize autonomous vehicle behavior based on dispositional trust levels in CARLA simulation environment.

## Project Overview
The project focuses on adapting autonomous vehicle behavior based on user's dispositional trust - the baseline level of trust a person has in automated systems. The system uses deep reinforcement learning to dynamically adjust driving behavior according to trust feedback. This project focuses on evaluating the effect of different deep reinforcement learning algorithms on the performance of the system.

## Prerequisites
- Ubuntu 20.04
- Python 3.7
- CARLA Simulator 0.9.15
- GPU with at least 8GB memory (recommended)

## Installation Steps

1. **Install CARLA Simulator**
```bash
# Download CARLA 0.9.15
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz

# Extract to your preferred location (e.g., /opt/carla-simulator)
sudo mkdir -p /opt/carla-simulator
sudo tar -xf CARLA_0.9.15.tar.gz -C /opt/carla-simulator

# Add CARLA Python API to PYTHONPATH
echo 'export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg' >> ~/.bashrc
source ~/.bashrc
```

2. **Set up Python Environment**
```bash
# Create and activate virtual environment
python3.7 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Verify CARLA Installation**
```bash
# Start CARLA simulator
cd /opt/carla-simulator
./CarlaUE4.sh -quality-level=Low
```

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
└── tests/            # Test cases
```

## Running Scenarios
1. Start CARLA simulator
2. Activate virtual environment
3. Run scenarios:
```bash
python src/main.py --scenario [scenario_name]
```

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
5. Pedestrian Interaction

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Generate Traffic Scene

```bash
python3 simulation/generate_traffic_scene.py
```
