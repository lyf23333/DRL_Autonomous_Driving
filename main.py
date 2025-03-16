import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment import CarlaEnv, CarlaEnvDiscrete
from src.trust.trust_interface import TrustInterface
from src.agents.drl_agent import DRLAgent
from scenarios import LaneSwitchingScenario, UrbanTrafficScenario, ObstacleAvoidanceScenario
from src.utils.carla_server import CarlaServerManager

def parse_args():
    parser = argparse.ArgumentParser(description='DRL Autonomous Driving with Trust Adaptation')
    
    # Scenario and algorithm options
    parser.add_argument('--scenario', type=str, default='lane_switching',
                      choices=['lane_switching', 'urban_traffic', 'obstacle_avoidance'],
                      help='Scenario to run')
    parser.add_argument('--algorithm', type=str, default='ppo',
                      choices=['ppo', 'sac', 'ddpg', 'dqn'],
                      help='DRL algorithm to use')
    
    # Training/evaluation options
    parser.add_argument('--train', action='store_true',
                      help='Train the agent')
    parser.add_argument('--eval', action='store_true',
                      help='Evaluate the agent')
    parser.add_argument('--render', action='store_true',
                      help='Rendering mode')
    
    # CARLA server options
    parser.add_argument('--start-carla', action='store_true',
                      help='Start CARLA server automatically')
    parser.add_argument('--carla-path', type=str, default=None,
                      help='Path to CARLA installation (if not set, will try to auto-detect)')
    parser.add_argument('--port', type=int, default=2000,
                      help='Port to run CARLA server on')
    parser.add_argument('--town', type=str, default='Town01',
                      help='CARLA town/map to use')
    parser.add_argument('--quality', type=str, default='Epic', choices=['Low', 'Epic'],
                      help='Graphics quality for CARLA')
    parser.add_argument('--offscreen', action='store_true',
                      help='Run CARLA in offscreen mode (no rendering)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Start CARLA server if requested
    carla_server = None
    if args.start_carla:
        print("Starting CARLA server...")
        carla_server = CarlaServerManager()
        success = carla_server.start_server(
            port=args.port,
            town=args.town,
            quality=args.quality,
            offscreen=args.offscreen,
            carla_path=args.carla_path
        )
        
        if not success:
            print("Failed to start CARLA server. Exiting.")
            sys.exit(1)
    
    # Initialize environment with configurable parameters
    if args.algorithm == 'dqn':
        env = CarlaEnvDiscrete(
            trust_interface=TrustInterface(), 
            town=args.town,
            port=args.port,
            render_mode=args.render
        )
    else:
        env = CarlaEnv(
            trust_interface=TrustInterface(), 
            town=args.town,
            port=args.port,
            render_mode=args.render
        )
    
    # Initialize DRL agent
    agent = DRLAgent(
        env=env,
        algorithm=args.algorithm,
    )
    
    # Load appropriate scenario
    scenario_map = {
        'lane_switching': LaneSwitchingScenario,
        'urban_traffic': UrbanTrafficScenario,
        'obstacle_avoidance': ObstacleAvoidanceScenario
    }
    
    scenario_class = scenario_map.get(args.scenario)
    if scenario_class is None:
        print(f"Error: Unknown scenario {args.scenario}")
        sys.exit(1)
    
    # TODO: We need to consider whether we train one for each agent or we train a single agent for all scenarios
    scenario = scenario_class(env)
    
    try:
        if args.train:
            # Training loop
            agent.train(scenario)
        elif args.eval:
            # Evaluation loop
            agent.evaluate(scenario)
        else:
            print("Please specify either --train or --eval")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Cleanup
        env.close()
        
        # Stop CARLA server if we started it
        if carla_server:
            carla_server.stop_server()

if __name__ == "__main__":
    main() 