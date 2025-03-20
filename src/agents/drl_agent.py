from stable_baselines3 import PPO, SAC, DDPG, DQN
import numpy as np
import os
from typing import Type, Optional
from torch.utils.tensorboard import SummaryWriter
import time


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
        
        # Initialize the appropriate algorithm
        self.model = self._create_model()
        
    def _create_model(self):
        """Create the DRL model based on specified algorithm"""
        if self.algorithm == 'ppo':
            return PPO(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                n_epochs=5,
                n_steps=512,
            )
        elif self.algorithm == 'sac':
            return SAC(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        elif self.algorithm == 'ddpg':
            return DDPG(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        elif self.algorithm == 'dqn':
            return DQN(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, scenario, total_timesteps=100000, scenario_config=None):
        """Train the agent on a specific scenario"""
        # Set scenario in environment
        self.env.set_scenario(scenario, scenario_config)
        
        # Initialize tensorboard writer
        run_name = f"{self.algorithm}_{scenario.__class__.__name__}_{int(time.time())}"
        self.tb_writer = SummaryWriter(os.path.join(self.tensorboard_log, run_name))
        
        # Custom callback function for logging
        def custom_callback(locals, globals):
            """Custom callback for logging during training"""
            # Get current step
            step = locals.get('self').num_timesteps
            
            # Get the current info
            infos = locals.get('infos', [{}])
            if len(infos) > 0:
                info = infos[0]  # Get the first environment's info
                
                # Track episode reward accumulation
                if locals.get('dones')[0]:
                    # Calculate episode reward when episode ends
                    episode_reward = locals.get('episode').get('r', 0)
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
                
            return True
            
        try:
            # Start training with custom callback
            self.model.learn(total_timesteps=total_timesteps, callback=custom_callback)
            
            # Save the trained model
            model_path = os.path.join(
                self.models_dir,
                f"{self.algorithm}_{scenario.__class__.__name__}.zip"
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
    
    def evaluate(self, scenario_class: Type, n_episodes=10, scenario_config=None):
        """Evaluate the agent on a specific scenario"""
        # Create and setup scenario
        scenario = scenario_class(self.env)
        self.env.set_scenario(scenario, scenario_config)
        
        # Initialize tensorboard writer for evaluation
        run_name = f"eval_{self.algorithm}_{scenario.__class__.__name__}_{int(time.time())}"
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
        """Load a trained model"""
        if self.algorithm == 'ppo':
            self.model = PPO.load(model_path, env=self.env)
        elif self.algorithm == 'sac':
            self.model = SAC.load(model_path, env=self.env)
        elif self.algorithm == 'ddpg':
            self.model = DDPG.load(model_path, env=self.env)
        elif self.algorithm == 'dqn':
            self.model = DQN.load(model_path, env=self.env)
        else:
            raise ValueError(f"Unsupported algorithm for loading: {self.algorithm}") 