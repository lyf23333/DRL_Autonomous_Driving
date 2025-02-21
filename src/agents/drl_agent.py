from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

"""
This file contains the implementation of the DRL agent.
Note: the drl part is not implemented and tested yet.
"""

class TrustCallback(BaseCallback):
    """Custom callback for tracking trust-related metrics during training"""
    
    def __init__(self, trust_interface, verbose=0):
        super(TrustCallback, self).__init__(verbose)
        self.trust_interface = trust_interface
        
    def _on_step(self):
        """Update policy based on current trust state"""
        trust_state = self.trust_interface.get_current_trust_state()
        # Implement trust-based policy adaptation here
        return True

class DRLAgent:
    def __init__(self, env, algorithm='ppo', trust_interface=None):
        self.env = env
        self.trust_interface = trust_interface
        self.algorithm = algorithm.lower()
        
        # Set up model paths
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize the appropriate algorithm
        self.model = self._create_model()
        
    def _create_model(self):
        """Create the DRL model based on specified algorithm"""
        if self.algorithm == 'ppo':
            return PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log="./tensorboard/"
            )
        elif self.algorithm == 'sac':
            return SAC(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log="./tensorboard/"
            )
        elif self.algorithm == 'ddpg':
            return DDPG(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log="./tensorboard/"
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, scenario, total_timesteps=100000):
        """Train the agent on a specific scenario"""
        # Setup scenario
        scenario.setup()
        
        # Create callback for trust adaptation
        callback = TrustCallback(self.trust_interface)
        
        try:
            # Start training
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback
            )
            
            # Save the trained model
            model_path = os.path.join(
                self.models_dir,
                f"{self.algorithm}_{scenario.__class__.__name__}.zip"
            )
            self.model.save(model_path)
            
        finally:
            # Cleanup scenario
            scenario.cleanup()
    
    def evaluate(self, scenario, n_episodes=10):
        """Evaluate the agent on a specific scenario"""
        # Setup scenario
        scenario.setup()
        
        try:
            total_reward = 0
            trust_levels = []
            
            for episode in range(n_episodes):
                obs = self.env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # Get action from model
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Execute action
                    obs, reward, done, info = self.env.step(action)
                    
                    # Update metrics
                    episode_reward += reward
                    trust_levels.append(info['trust_level'])
                    
                    # Update trust interface display
                    if self.trust_interface:
                        self.trust_interface.update_display()
                
                total_reward += episode_reward
                print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}")
            
            # Print evaluation results
            print("\nEvaluation Results:")
            print(f"Average Reward: {total_reward / n_episodes:.2f}")
            print(f"Average Trust Level: {np.mean(trust_levels):.2f}")
            
        finally:
            # Cleanup scenario
            scenario.cleanup()
    
    def load(self, model_path):
        """Load a trained model"""
        if self.algorithm == 'ppo':
            self.model = PPO.load(model_path, env=self.env)
        elif self.algorithm == 'sac':
            self.model = SAC.load(model_path, env=self.env)
        elif self.algorithm == 'ddpg':
            self.model = DDPG.load(model_path, env=self.env) 