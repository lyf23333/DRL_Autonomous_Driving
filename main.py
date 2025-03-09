import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import CarlaEnv, CarlaEnvDiscrete
from trust.trust_interface import TrustInterface
from agents.drl_agent import DRLAgent
from scenarios import LaneSwitchingScenario, UrbanTrafficScenario, ObstacleAvoidanceScenario

def parse_args():
    parser = argparse.ArgumentParser(description='DRL Autonomous Driving with Trust Adaptation')
    parser.add_argument('--scenario', type=str, default='lane_switching',
                      choices=['lane_switching', 'urban_traffic', 'obstacle_avoidance'],
                      help='Scenario to run')
    parser.add_argument('--algorithm', type=str, default='ppo',
                      choices=['ppo', 'sac', 'ddpg', 'dqn'],
                      help='DRL algorithm to use')
    parser.add_argument('--train', action='store_true',
                      help='Train the agent')
    parser.add_argument('--eval', action='store_true',
                      help='Evaluate the agent')
    parser.add_argument('--render', action='store_true',
                      help='Rendering mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize environment
    if args.algorithm == 'dqn':
        env = CarlaEnvDiscrete(trust_interface=TrustInterface(), render_mode=args.render)
    else:
        env = CarlaEnv(trust_interface=TrustInterface(), render_mode=args.render)
    
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

if __name__ == '__main__':
    main() 