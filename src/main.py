import argparse
import os
import sys
from environment.carla_env import CarlaEnv
from trust.trust_interface import TrustInterface
from agents.drl_agent import DRLAgent
from scenarios.lane_switching import LaneSwitchingScenario
from scenarios.urban_traffic import UrbanTrafficScenario
from scenarios.obstacle_avoidance import ObstacleAvoidanceScenario

def parse_args():
    parser = argparse.ArgumentParser(description='DRL Autonomous Driving with Trust Adaptation')
    parser.add_argument('--scenario', type=str, default='lane_switching',
                      choices=['lane_switching', 'urban_traffic', 'obstacle_avoidance'],
                      help='Scenario to run')
    parser.add_argument('--algorithm', type=str, default='ppo',
                      choices=['ppo', 'sac', 'ddpg'],
                      help='DRL algorithm to use')
    parser.add_argument('--train', action='store_true',
                      help='Train the agent')
    parser.add_argument('--eval', action='store_true',
                      help='Evaluate the agent')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize environment
    env = CarlaEnv()
    
    # Initialize trust interface
    trust_interface = TrustInterface()
    
    # Initialize DRL agent
    agent = DRLAgent(
        env=env,
        algorithm=args.algorithm,
        trust_interface=trust_interface
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
        trust_interface.cleanup()

if __name__ == '__main__':
    main() 