import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment import CarlaEnv, CarlaEnvDiscrete
from src.environment.carla_env_config import CarlaEnvConfig
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
    parser.add_argument('--timesteps', type=int, default=100000,
                      help='Total timesteps for training')
    parser.add_argument('--run-name', type=str, default=None,
                      help='Custom name for this run, used in saved files and logs')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Initial learning rate')
    parser.add_argument('--lr-schedule', type=str, default='exponential',
                      choices=['constant', 'linear', 'exponential', 'cosine'],
                      help='Learning rate schedule')
    parser.add_argument('--lr-decay-factor', type=float, default=0.05,
                      help='Factor by which to decay learning rate (for exponential)')
    
    # Model loading and checkpointing options
    parser.add_argument('--load-model', type=str, default=None,
                      help='Path to a trained model to load')
    parser.add_argument('--checkpoint-freq', type=int, default=100000,
                      help='Save checkpoint every N timesteps')
    parser.add_argument('--resume-training', action='store_true',
                      help='Resume training from the loaded model')
    
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
    env_config = CarlaEnvConfig.from_json('configs/default_config.json')
    env_config.town = args.town
    env_config.port = args.port
    env_config.render_mode = args.render
    
    if args.algorithm == 'dqn':
        env = CarlaEnvDiscrete(
            trust_interface=TrustInterface(), 
            config=env_config,
            eval=args.eval
        )
    else:
        env = CarlaEnv(
            trust_interface=TrustInterface(), 
            config=env_config,
            eval=args.eval
        )
    
    # Initialize DRL agent
    agent = DRLAgent(
        env=env,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps
    )
    
    # Set learning rate parameters
    agent.set_learning_rate_params(
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_decay_factor=args.lr_decay_factor,
        total_timesteps=args.timesteps
    )
    
    # Load a pre-trained model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent.load(args.load_model)
    
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
            # Set checkpoint frequency if provided
            agent.checkpoint_freq = args.checkpoint_freq
            
            # Training loop
            if args.resume_training and args.load_model:
                print(f"Resuming training from {args.load_model}")
                # When resuming, we need to reset the environment with the loaded policy
                env.reset()
            
            agent.train(scenario, total_timesteps=args.timesteps, run_name=args.run_name)
        elif args.eval:
            # Evaluation loop
            agent.evaluate(scenario_class, n_episodes=10, run_name=args.run_name)
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