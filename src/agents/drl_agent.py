from stable_baselines3 import PPO, SAC, DDPG, DQN
import numpy as np
import os
from typing import Type, Optional
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import math

class DRLAgent:
    def __init__(self, env, algorithm='ppo'):
        self.env = env
        self.algorithm = algorithm.lower()
        
        # Set up model paths
        self.models_dir = os.path.join("models", self.algorithm)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set up tensorboard
        self.tensorboard_log = "./tensorboard/"
        os.makedirs(self.tensorboard_log, exist_ok=True)
        self.tb_writer = None
        
        # Default checkpoint frequency
        self.checkpoint_freq = 10000  # Save every 10k timesteps
        
        # Learning rate settings (defaults)
        self.learning_rate = 3e-4
        self.lr_schedule = 'constant'
        self.lr_decay_factor = 0.1
        
        # Initialize the appropriate algorithm
        self.model = self._create_model()
        
    def _get_learning_rate_schedule(self, total_timesteps):
        """Create a learning rate schedule function based on the specified schedule type"""
        initial_lr = self.learning_rate
        
        if self.lr_schedule == 'constant':
            # Constant learning rate
            return lambda step: initial_lr
            
        elif self.lr_schedule == 'linear':
            # Linear decay to 10% of initial learning rate
            def linear_schedule(step):
                fraction = 1.0 - (step / total_timesteps)
                return initial_lr * max(0.1, fraction)
            return linear_schedule
            
        elif self.lr_schedule == 'exponential':
            # Exponential decay
            decay_factor = self.lr_decay_factor
            def exponential_schedule(step):
                # Calculate step fraction (0 to 1)
                fraction = step / total_timesteps
                # Decay learning rate exponentially
                return initial_lr * (decay_factor ** fraction)
            return exponential_schedule
            
        elif self.lr_schedule == 'cosine':
            # Cosine annealing
            def cosine_schedule(step):
                fraction = min(1.0, step / total_timesteps)
                return initial_lr * 0.1 + 0.9 * initial_lr * (1 + math.cos(math.pi * fraction)) / 2
            return cosine_schedule
            
        else:
            # Default to constant if invalid schedule type
            return lambda step: initial_lr
        
    def _create_model(self):
        """Create the DRL model based on specified algorithm"""
        if self.algorithm == 'ppo':
            return PPO(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=self.learning_rate,
            )
        elif self.algorithm == 'sac':
            return SAC(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=self.learning_rate,
            )
        elif self.algorithm == 'ddpg':
            return DDPG(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=self.learning_rate,
            )
        elif self.algorithm == 'dqn':
            return DQN(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=self.learning_rate,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def set_learning_rate_params(self, learning_rate, lr_schedule='constant', lr_decay_factor=0.1):
        """Set learning rate and schedule parameters
        
        Args:
            learning_rate: Initial learning rate
            lr_schedule: Learning rate schedule type ('constant', 'linear', 'exponential', 'cosine')
            lr_decay_factor: Factor for exponential decay
        """
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.lr_decay_factor = lr_decay_factor
        
        # Recreate the model with the new learning rate settings
        self.model = self._create_model()
    
    def train(self, scenario, total_timesteps=100000, scenario_config=None, run_name=None):
        """Train the agent on a specific scenario
        
        Args:
            scenario: The scenario to train on
            total_timesteps: Total number of timesteps to train for
            scenario_config: Optional configuration for the scenario
            run_name: Custom name for this training run, used in saved files and logs
        """
        # Set scenario in environment
        self.env.set_scenario(scenario, scenario_config)
        
        # Generate a timestamp for unique identification
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
        # Create run name with custom prefix if provided
        if run_name:
            run_name = f"{run_name}_{self.algorithm}_{scenario.__class__.__name__}_{timestamp}"
        else:
            run_name = f"{self.algorithm}_{scenario.__class__.__name__}_{timestamp}"
        
        # Initialize tensorboard writer
        self.tb_writer = SummaryWriter(os.path.join(self.tensorboard_log, run_name))
        
        # Create directory for checkpoints
        checkpoints_dir = os.path.join(self.models_dir, "checkpoints", run_name)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Set up learning rate schedule
        lr_schedule = self._get_learning_rate_schedule(total_timesteps)
        
        # Custom callback function for logging and checkpointing
        def custom_callback(locals, globals):
            """Custom callback for logging during training"""
            # Get current step
            step = locals.get('self').num_timesteps
            
            # Update learning rate based on schedule
            if self.lr_schedule != 'constant':
                new_lr = lr_schedule(step)
                if hasattr(locals.get('self'), 'learning_rate'):
                    locals.get('self').learning_rate = new_lr
                self.tb_writer.add_scalar('train/learning_rate', new_lr, step)
            
            # Save checkpoint every checkpoint_freq steps
            if step > 0 and step % self.checkpoint_freq == 0:
                checkpoint_path = os.path.join(
                    checkpoints_dir,
                    f"{run_name}_steps_{step}.zip"
                )
                self.model.save(checkpoint_path)
                print(f"\nCheckpoint saved at step {step}: {checkpoint_path}\n")
            
            # Get the current info
            infos = locals.get('infos', [{}])
            if len(infos) > 0:
                info = infos[0]  # Get the first environment's info
                
                # Track episode reward accumulation
                if locals.get('dones')[0]:
                    # Calculate episode reward when episode ends
                    episode_reward = locals.get('rewards', 0)
                    info['episode_reward'] = episode_reward
                    
                    # Log episode-level metrics
                    self.tb_writer.add_scalar('Reward/episode_reward', episode_reward, step)
                
                # Always log step-level metrics
                # Log trust level
                self.tb_writer.add_scalar('Metrics/trust_level', info.get('trust_level', 0), step)
                
                # Log driving metrics
                driving_metrics = info.get('driving_metrics', {})
                for metric, value in driving_metrics.items():
                    self.tb_writer.add_scalar(f'Metrics/{metric}', value, step)
                    
                # Log reward components
                reward_components = info.get('reward_components', {})
                for component, value in reward_components.items():
                    self.tb_writer.add_scalar(f'Reward/component_{component}', value, step)
                
                # Log speed metrics
                self.tb_writer.add_scalar('Metrics/current_speed', info.get('current_speed', 0), step)
                self.tb_writer.add_scalar('Metrics/target_speed', info.get('target_speed', 0), step)
                
                # Log intervention data
                self.tb_writer.add_scalar('Metrics/intervention_probability', 
                                        info.get('intervention_probability', 0), step)
                self.tb_writer.add_scalar('Metrics/intervention_active', 
                                        1.0 if info.get('intervention_active', False) else 0.0, step)
                
                # Log waypoint progress
                self.tb_writer.add_scalar('Metrics/current_waypoint_idx', 
                                         info.get('current_waypoint_idx', 0), step)
                self.tb_writer.add_scalar('Metrics/waypoints_remaining', 
                                         info.get('waypoints_remaining', 0), step)
                
            return True
            
        try:
            # Start training with custom callback
            self.model.learn(total_timesteps=total_timesteps, callback=custom_callback)
            
            # Save the trained model
            model_path = os.path.join(
                self.models_dir,
                f"{run_name}.zip"
            )
            self.model.save(model_path)
            print(f"Model successfully saved to {model_path}")
            
        except Exception as e:
            import traceback
            print(f"Exception during training: {type(e).__name__}: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("Training terminated early due to error.")
            
        finally:
            # Close tensorboard writer
            if self.tb_writer:
                self.tb_writer.close()
                self.tb_writer = None
                
            # Cleanup scenario
            print("Cleaning up scenario resources...")
            if hasattr(self.env, 'active_scenario'):
                self.env.active_scenario.cleanup()
            print("Cleanup complete.")
    
    def evaluate(self, scenario_class: Type, n_episodes=10, scenario_config=None, run_name=None):
        """Evaluate the agent on a specific scenario
        
        Args:
            scenario_class: The scenario class to evaluate on
            n_episodes: Number of episodes to evaluate
            scenario_config: Optional configuration for the scenario
            run_name: Custom name for this evaluation run, used in logs
        """
        # Create and setup scenario
        scenario = scenario_class(self.env)
        self.env.set_scenario(scenario, scenario_config)
        
        # Generate a timestamp for unique identification
        timestamp = int(time.time())
        
        # Create run name with custom prefix if provided
        if run_name:
            run_name = f"eval_{run_name}_{self.algorithm}_{scenario.__class__.__name__}_{timestamp}"
        else:
            run_name = f"eval_{self.algorithm}_{scenario.__class__.__name__}_{timestamp}"
        
        # Initialize tensorboard writer for evaluation
        self.tb_writer = SummaryWriter(os.path.join(self.tensorboard_log, run_name))
        step = 0
        episode_step = 0
        
        try:
            total_reward = 0
            trust_levels = []
            completion_rate = 0
            
            for episode in range(n_episodes):
                obs, info = self.env.reset()
                done = False
                truncated = False
                episode_reward = 0
                episode_step = 0
                
                while not (done or truncated):
                    # Get action from model
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Execute action
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    # Update metrics
                    episode_reward += reward
                    trust_levels.append(info['trust_level'])
                    
                    # Log per-step metrics
                    # Log trust level
                    self.tb_writer.add_scalar('Metrics/trust_level', info.get('trust_level', 0), step)
                    
                    # Log driving metrics
                    driving_metrics = info.get('driving_metrics', {})
                    for metric, value in driving_metrics.items():
                        self.tb_writer.add_scalar(f'Metrics/{metric}', value, step)
                        
                    # Log reward components
                    reward_components = info.get('reward_components', {})
                    for component, value in reward_components.items():
                        self.tb_writer.add_scalar(f'Reward/component_{component}', value, step)
                    
                    # Log instant reward
                    self.tb_writer.add_scalar('Reward/instant_reward', reward, step)
                    
                    # Log speed metrics
                    self.tb_writer.add_scalar('Metrics/current_speed', info.get('current_speed', 0), step)
                    self.tb_writer.add_scalar('Metrics/target_speed', info.get('target_speed', 0), step)
                    
                    # Log intervention data
                    self.tb_writer.add_scalar('Metrics/intervention_probability', 
                                             info.get('intervention_probability', 0), step)
                    self.tb_writer.add_scalar('Metrics/intervention_active', 
                                             1.0 if info.get('intervention_active', False) else 0.0, step)
                    
                    # Log waypoint progress
                    self.tb_writer.add_scalar('Metrics/current_waypoint_idx', 
                                             info.get('current_waypoint_idx', 0), step)
                    self.tb_writer.add_scalar('Metrics/waypoints_remaining', 
                                             info.get('waypoints_remaining', 0), step)
                    
                    # Increment steps
                    step += 1
                    episode_step += 1
                    
                    # Check scenario completion
                    if info.get('scenario_complete', False):
                        completion_rate += 1
                
                # Log episode-level metrics
                self.tb_writer.add_scalar('Reward/episode_reward', episode_reward, episode)
                self.tb_writer.add_scalar('Metrics/episode_length', episode_step, episode)
                
                total_reward += episode_reward
                print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Steps = {episode_step}")
            
            # Print evaluation results
            print("\nEvaluation Results:")
            print(f"Average Reward: {total_reward / n_episodes:.2f}")
            print(f"Average Trust Level: {np.mean(trust_levels):.2f}")
            print(f"Scenario Completion Rate: {completion_rate / n_episodes * 100:.1f}%")
            
        except Exception as e:
            import traceback
            print(f"Exception during evaluation: {type(e).__name__}: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("Evaluation terminated early due to error.")
            
        finally:
            # Close tensorboard writer
            if self.tb_writer:
                self.tb_writer.close()
                self.tb_writer = None
                
            # Cleanup scenario
            print("Cleaning up scenario resources...")
            if hasattr(self.env, 'active_scenario'):
                self.env.active_scenario.cleanup()
            print("Cleanup complete.")
    
    def load(self, model_path):
        """Load a trained model from a file
        
        This method can be used to load either a full model or a checkpoint saved during training.
        
        Example usage:
            # Load the final model
            agent.load("models/ppo/ppo_ObstacleAvoidanceScenario.zip")
            
            # Load a checkpoint
            agent.load("models/ppo/checkpoints/ppo_ObstacleAvoidanceScenario_1650123456/ppo_ObstacleAvoidanceScenario_steps_100000.zip")
        
        Args:
            model_path (str): Path to the saved model file (.zip)
        """
        try:
            # Determine algorithm type from the model path
            if 'ppo' in model_path.lower():
                self.model = PPO.load(model_path, env=self.env)
                algorithm = 'ppo'
            elif 'sac' in model_path.lower():
                self.model = SAC.load(model_path, env=self.env)
                algorithm = 'sac'
            elif 'ddpg' in model_path.lower():
                self.model = DDPG.load(model_path, env=self.env)
                algorithm = 'ddpg'
            elif 'dqn' in model_path.lower():
                self.model = DQN.load(model_path, env=self.env)
                algorithm = 'dqn'
            else:
                # Default to the algorithm specified in the constructor
                if self.algorithm == 'ppo':
                    self.model = PPO.load(model_path, env=self.env)
                elif self.algorithm == 'sac':
                    self.model = SAC.load(model_path, env=self.env)
                elif self.algorithm == 'ddpg':
                    self.model = DDPG.load(model_path, env=self.env)
                elif self.algorithm == 'dqn':
                    self.model = DQN.load(model_path, env=self.env)
                algorithm = self.algorithm
                
            print(f"Successfully loaded {algorithm.upper()} model from {model_path}")
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise 