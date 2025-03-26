from stable_baselines3 import PPO, SAC, DDPG, DQN
import numpy as np
import os
from typing import Type, Optional
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import math
class DRLAgent:
    def __init__(self, env, algorithm='ppo', total_timesteps=100000):
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
        self.model = self._create_model(total_timesteps)
        
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
        
    def _create_model(self, total_timesteps):
        """Create the DRL model based on specified algorithm"""
        sched_LR = self._get_learning_rate_schedule(total_timesteps) # learning_rate = sched_LR.value
        if self.algorithm == 'ppo':
            return PPO(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=sched_LR.value,
            )
        elif self.algorithm == 'sac':
            return SAC(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=sched_LR.value,
            )
        elif self.algorithm == 'ddpg':
            return DDPG(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=sched_LR.value,
            )
        elif self.algorithm == 'dqn':
            return DQN(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.tensorboard_log,
                learning_rate=sched_LR.value,
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
            run_name = f"{timestamp}_{run_name}_{self.algorithm}_{scenario.__class__.__name__}"
        else:
            run_name = f"{timestamp}_{self.algorithm}_{scenario.__class__.__name__}"
        
        # Initialize tensorboard writer
        self.tb_writer = SummaryWriter(os.path.join(self.tensorboard_log, run_name))
        
        # Create directory for checkpoints
        checkpoints_dir = os.path.join(self.models_dir, "checkpoints", run_name)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Set up learning rate schedule
        lr_schedule = self._get_learning_rate_schedule(total_timesteps)
        
        # Track performance metrics across episodes
        episode_count = 0
        total_interventions = 0
        total_crashes = 0
        total_traffic_violations = 0
        engagement_durations = []  # Time periods system was engaged without intervention
        episode_rewards = []
        intervention_rates = []
        
        # Custom callback function for logging and checkpointing
        def custom_callback(locals, globals):
            """Custom callback for logging during training"""
            nonlocal episode_count, total_interventions, total_crashes, total_traffic_violations
            nonlocal engagement_durations, episode_rewards, intervention_rates
            
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
                    f"checkpoint_{step}.zip"
                )
                self.model.save(checkpoint_path)
                print(f"\nCheckpoint saved at step {step}: {checkpoint_path}\n")
            
            # Get the current info
            infos = locals.get('infos', [{}])
            if len(infos) > 0:
                info = infos[0]  # Get the first environment's info
                
                # Track episode completion
                dones = locals.get('dones', [False])
                if dones[0]:
                    # Increment episode counter
                    episode_count += 1
                    
                    # Calculate episode reward when episode ends
                    episode_reward = sum(locals.get('rewards', [0]))
                    episode_rewards.append(episode_reward)
                    
                    # Log episode-level metrics
                    self.tb_writer.add_scalar('Reward/episode_reward', episode_reward, step)
                    self.tb_writer.add_scalar('Metrics/episode_count', episode_count, step)
                    
                    # Calculate and log average metrics every 10 episodes
                    if episode_count % 10 == 0 and episode_count > 0:
                        # Calculate average reward
                        avg_reward = sum(episode_rewards[-10:]) / 10
                        self.tb_writer.add_scalar('Analysis/avg_episode_reward_10', avg_reward, step)
                        
                        # Calculate crash rate per episode
                        crash_rate = total_crashes / episode_count
                        self.tb_writer.add_scalar('Analysis/crash_rate', crash_rate, step)
                        
                        # Calculate intervention rate per episode
                        if len(intervention_rates) > 0:
                            avg_intervention_rate = sum(intervention_rates[-10:]) / min(10, len(intervention_rates))
                            self.tb_writer.add_scalar('Analysis/avg_intervention_rate_10', avg_intervention_rate, step)
                        
                        # Traffic violation rate
                        violation_rate = total_traffic_violations / episode_count
                        self.tb_writer.add_scalar('Analysis/traffic_violation_rate', violation_rate, step)
                
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
                is_intervention = info.get('intervention_active', False)
                intervention_prob = info.get('intervention_probability', 0)
                self.tb_writer.add_scalar('Trust/intervention_probability', intervention_prob, step)
                self.tb_writer.add_scalar('Trust/intervention_active', 1.0 if is_intervention else 0.0, step)
                
                # Track intervention events
                if is_intervention:
                    total_interventions += 1
                    # Record this intervention
                    self.tb_writer.add_scalar('Trust/total_interventions', total_interventions, step)
                
                # Track crash events
                if info.get('collision_detected', False):
                    total_crashes += 1
                    self.tb_writer.add_scalar('Performance/total_crashes', total_crashes, step)
                    self.tb_writer.add_scalar('Performance/crash_event', 1.0, step)
                else:
                    self.tb_writer.add_scalar('Performance/crash_event', 0.0, step)
                
                # Track traffic violations (e.g., crossing red light, lane violations)
                traffic_violation = info.get('traffic_violation', False)
                if traffic_violation:
                    total_traffic_violations += 1
                    self.tb_writer.add_scalar('Performance/traffic_violation_event', 1.0, step)
                else:
                    self.tb_writer.add_scalar('Performance/traffic_violation_event', 0.0, step)
                self.tb_writer.add_scalar('Performance/total_traffic_violations', total_traffic_violations, step)
                
                # Calculate engagement metrics if episode ended
                if dones[0] and 'engagement_duration' in info:
                    engagement_durations.append(info['engagement_duration'])
                    avg_engagement = sum(engagement_durations) / len(engagement_durations)
                    self.tb_writer.add_scalar('Trust/avg_engagement_duration', avg_engagement, step)
                    self.tb_writer.add_scalar('Trust/episode_intervention_rate', 
                                             info.get('episode_intervention_count', 0) / max(1, info.get('episode_steps', 1)), 
                                             step)
                    intervention_rates.append(info.get('episode_intervention_count', 0) / max(1, info.get('episode_steps', 1)))
                
                # Record the speed adaptation ratio (how well the agent adapts to the target speed)
                if info.get('target_speed', 0) > 0:
                    speed_adaptation = info.get('current_speed', 0) / info.get('target_speed', 1)
                    self.tb_writer.add_scalar('Performance/speed_adaptation_ratio', speed_adaptation, step)
                
                # Log path following accuracy
                if 'waypoint_deviation' in info:
                    self.tb_writer.add_scalar('Performance/waypoint_deviation', info['waypoint_deviation'], step)
                
                # Log waypoint progress
                self.tb_writer.add_scalar('Progress/current_waypoint_idx', 
                                         info.get('current_waypoint_idx', 0), step)
                self.tb_writer.add_scalar('Progress/waypoints_remaining', 
                                         info.get('waypoints_remaining', 0), step)
                
                # Calculate trust calibration metrics
                if 'trust_level' in info and 'intervention_probability' in info:
                    # Trust-intervention relationship (how well trust aligns with intervention needs)
                    trust_calibration = abs(info['trust_level'] - (1 - info['intervention_probability']))
                    self.tb_writer.add_scalar('Trust/calibration_error', trust_calibration, step)
                
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
            
            # Log final performance statistics
            if episode_count > 0:
                print("\nTraining Performance Summary:")
                print(f"Total Episodes: {episode_count}")
                print(f"Average Episode Reward: {sum(episode_rewards) / episode_count:.2f}")
                print(f"Total Manual Interventions: {total_interventions}")
                # print(f"Intervention Rate: {total_interventions / max(1, sum(info.get('episode_steps', 0) for info in episode_rewards)):.4f}")
                print(f"Crash Rate: {total_crashes / episode_count:.4f}")
                print(f"Traffic Violation Rate: {total_traffic_violations / episode_count:.4f}")
                
                if engagement_durations:
                    print(f"Average Engagement Duration: {sum(engagement_durations) / len(engagement_durations):.2f} steps")
            
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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create run name with custom prefix if provided
        if run_name:
            run_name = f"eval_{run_name}_{self.algorithm}_{scenario.__class__.__name__}_{timestamp}"
        else:
            run_name = f"eval_{self.algorithm}_{scenario.__class__.__name__}_{timestamp}"
        
        # Initialize tensorboard writer for evaluation
        self.tb_writer = SummaryWriter(os.path.join(self.tensorboard_log, run_name))
        step = 0
        episode_step = 0
        
        # Track performance metrics
        total_interventions = 0
        total_crashes = 0
        total_traffic_violations = 0
        engagement_durations = []
        episode_rewards = []
        intervention_rates = []
        waypoint_completions = []
        
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
                episode_interventions = 0
                last_intervention_step = -1
                engagement_start = 0
                
                # Episode tracking metrics
                episode_collisions = 0
                episode_traffic_violations = 0
                
                while not (done or truncated):
                    # Get action from model
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Execute action
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    # Update metrics
                    episode_reward += reward
                    trust_levels.append(info['trust_level'])
                    
                    # Count steps since last intervention (engagement duration)
                    if not info.get('intervention_active', False):
                        if last_intervention_step == episode_step - 1:
                            # New engagement period starts
                            engagement_start = episode_step
                    else:
                        # Intervention occurred
                        episode_interventions += 1
                        total_interventions += 1
                        
                        # Record engagement duration if applicable
                        if last_intervention_step < engagement_start:
                            current_engagement = episode_step - engagement_start
                            if current_engagement > 0:
                                engagement_durations.append(current_engagement)
                        
                        last_intervention_step = episode_step
                    
                    # Detect collision events
                    if info.get('collision_detected', False):
                        episode_collisions += 1
                        total_crashes += 1
                    
                    # Detect traffic violations
                    if info.get('traffic_violation', False):
                        episode_traffic_violations += 1
                        total_traffic_violations += 1
                    
                    # Log per-step metrics
                    # Log trust calibration metrics
                    self.tb_writer.add_scalar('Trust/trust_level', info.get('trust_level', 0), step)
                    self.tb_writer.add_scalar('Trust/intervention_probability', info.get('intervention_probability', 0), step)
                    self.tb_writer.add_scalar('Trust/intervention_active', 1.0 if info.get('intervention_active', False) else 0.0, step)
                    
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
                    
                    # Log performance metrics
                    self.tb_writer.add_scalar('Performance/crash_event', 1.0 if info.get('collision_detected', False) else 0.0, step)
                    self.tb_writer.add_scalar('Performance/traffic_violation_event', 1.0 if info.get('traffic_violation', False) else 0.0, step)
                    
                    # Calculate speed adaptation ratio
                    if info.get('target_speed', 0) > 0:
                        speed_adaptation = info.get('current_speed', 0) / info.get('target_speed', 1)
                        self.tb_writer.add_scalar('Performance/speed_adaptation_ratio', speed_adaptation, step)
                    
                    # Log path following metrics
                    self.tb_writer.add_scalar('Progress/current_waypoint_idx', info.get('current_waypoint_idx', 0), step)
                    self.tb_writer.add_scalar('Progress/waypoints_remaining', info.get('waypoints_remaining', 0), step)
                    
                    if 'waypoint_deviation' in info:
                        self.tb_writer.add_scalar('Performance/waypoint_deviation', info['waypoint_deviation'], step)
                    
                    # Calculate trust calibration error
                    if 'trust_level' in info and 'intervention_probability' in info:
                        trust_calibration = abs(info['trust_level'] - (1 - info['intervention_probability']))
                        self.tb_writer.add_scalar('Trust/calibration_error', trust_calibration, step)
                    
                    # Increment steps
                    step += 1
                    episode_step += 1
                    
                    # Check scenario completion
                    if info.get('scenario_complete', False):
                        completion_rate += 1
                
                # Record final engagement duration if we end without intervention
                if last_intervention_step < engagement_start:
                    current_engagement = episode_step - engagement_start
                    if current_engagement > 0:
                        engagement_durations.append(current_engagement)
                
                # Log episode-level metrics
                self.tb_writer.add_scalar('Reward/episode_reward', episode_reward, episode)
                self.tb_writer.add_scalar('Metrics/episode_length', episode_step, episode)
                self.tb_writer.add_scalar('Trust/episode_interventions', episode_interventions, episode)
                self.tb_writer.add_scalar('Performance/episode_collisions', episode_collisions, episode)
                self.tb_writer.add_scalar('Performance/episode_traffic_violations', episode_traffic_violations, episode)
                
                # Calculate intervention rate for this episode
                if episode_step > 0:
                    intervention_rate = episode_interventions / episode_step
                    intervention_rates.append(intervention_rate)
                    self.tb_writer.add_scalar('Trust/episode_intervention_rate', intervention_rate, episode)
                
                # Track waypoint completion percentage
                if 'waypoints_total' in info and info['waypoints_total'] > 0:
                    waypoint_completion = info.get('current_waypoint_idx', 0) / info['waypoints_total']
                    waypoint_completions.append(waypoint_completion)
                    self.tb_writer.add_scalar('Progress/waypoint_completion', waypoint_completion, episode)
                
                # Add to running totals
                total_reward += episode_reward
                episode_rewards.append(episode_reward)
                
                print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Steps = {episode_step}, "
                      f"Interventions = {episode_interventions}, Crashes = {episode_collisions}")
            
            # Calculate and log aggregate metrics
            avg_reward = total_reward / n_episodes
            avg_trust = sum(trust_levels) / max(1, len(trust_levels))
            avg_intervention_rate = sum(intervention_rates) / max(1, len(intervention_rates))
            avg_engagement = sum(engagement_durations) / max(1, len(engagement_durations))
            
            self.tb_writer.add_scalar('Analysis/avg_episode_reward', avg_reward, 0)
            self.tb_writer.add_scalar('Analysis/avg_trust_level', avg_trust, 0)
            self.tb_writer.add_scalar('Analysis/completion_rate', completion_rate / n_episodes, 0)
            self.tb_writer.add_scalar('Analysis/crash_rate', total_crashes / n_episodes, 0)
            self.tb_writer.add_scalar('Analysis/avg_intervention_rate', avg_intervention_rate, 0)
            self.tb_writer.add_scalar('Analysis/avg_engagement_duration', avg_engagement, 0)
            
            # Print evaluation results
            print("\nEvaluation Results:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Trust Level: {avg_trust:.2f}")
            print(f"Scenario Completion Rate: {completion_rate / n_episodes * 100:.1f}%")
            print(f"Crash Rate: {total_crashes / n_episodes:.2f} crashes per episode")
            print(f"Average Intervention Rate: {avg_intervention_rate:.4f} interventions per step")
            print(f"Average Engagement Duration: {avg_engagement:.2f} steps")
            if waypoint_completions:
                avg_waypoint_completion = sum(waypoint_completions) / len(waypoint_completions)
                print(f"Average Waypoint Completion: {avg_waypoint_completion * 100:.1f}%")
            
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