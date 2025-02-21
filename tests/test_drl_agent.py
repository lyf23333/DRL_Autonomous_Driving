import pytest
import os
import numpy as np
from src.agents.drl_agent import DRLAgent, TrustCallback
from stable_baselines3 import PPO, SAC, DDPG

def test_agent_initialization(mock_env, trust_interface):
    """Test DRL agent initialization"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    assert agent.env == mock_env
    assert agent.trust_interface == trust_interface
    assert agent.algorithm == 'ppo'
    assert isinstance(agent.model, PPO)

def test_model_creation(mock_env, trust_interface):
    """Test model creation for different algorithms"""
    # Test PPO
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    assert isinstance(agent.model, PPO)
    
    # Test SAC
    agent = DRLAgent(mock_env, 'sac', trust_interface)
    assert isinstance(agent.model, SAC)
    
    # Test DDPG
    agent = DRLAgent(mock_env, 'ddpg', trust_interface)
    assert isinstance(agent.model, DDPG)
    
    # Test invalid algorithm
    with pytest.raises(ValueError):
        DRLAgent(mock_env, 'invalid_algo', trust_interface)

def test_trust_callback(mock_env, trust_interface):
    """Test trust callback functionality"""
    callback = TrustCallback(trust_interface)
    assert callback.trust_interface == trust_interface
    assert callback._on_step() is True  # Should always return True for now

def test_model_saving(mock_env, trust_interface, tmp_path):
    """Test model saving functionality"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    agent.models_dir = str(tmp_path / "models")
    os.makedirs(agent.models_dir, exist_ok=True)
    
    # Train for a minimal number of steps
    agent.train(mock_env, total_timesteps=1000)
    
    # Check if model file exists
    model_files = os.listdir(agent.models_dir)
    assert len(model_files) > 0
    assert any(f.endswith('.zip') for f in model_files)

def test_model_loading(mock_env, trust_interface, tmp_path):
    """Test model loading functionality"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    agent.models_dir = str(tmp_path / "models")
    os.makedirs(agent.models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(agent.models_dir, "test_model.zip")
    agent.model.save(model_path)
    
    # Load model
    agent.load(model_path)
    assert isinstance(agent.model, PPO)

def test_training_loop(mock_env, trust_interface, mocker):
    """Test training loop functionality"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    
    # Mock the learn method to avoid actual training
    mocker.patch.object(agent.model, 'learn')
    
    # Train for a small number of steps
    agent.train(mock_env, total_timesteps=1000)
    
    # Verify learn was called
    agent.model.learn.assert_called_once()

def test_evaluation(mock_env, trust_interface):
    """Test evaluation functionality"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    
    # Mock the predict method
    mock_env.reset.return_value = np.zeros(mock_env.observation_space.shape)
    mock_env.step.return_value = (
        np.zeros(mock_env.observation_space.shape),
        0.0,
        True,
        {'trust_level': 0.5, 'manual_interventions': 0}
    )
    
    # Run evaluation
    agent.evaluate(mock_env, n_episodes=2)
    
    # Verify reset and step were called
    assert mock_env.reset.called
    assert mock_env.step.called

def test_trust_integration(mock_env, trust_interface):
    """Test integration with trust interface"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    
    # Record an intervention
    trust_interface.record_intervention()
    
    # Train for a minimal number of steps
    agent.train(mock_env, total_timesteps=1000)
    
    # Verify trust level was updated
    assert trust_interface.trust_level < 0.5  # Initial trust level

def test_cleanup(mock_env, trust_interface):
    """Test cleanup functionality"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    
    # Train and evaluate
    agent.train(mock_env, total_timesteps=1000)
    agent.evaluate(mock_env, n_episodes=1)
    
    # Cleanup should not raise any errors
    try:
        mock_env.close()
        trust_interface.cleanup()
    except Exception as e:
        pytest.fail(f"Cleanup raised an exception: {e}")

def test_deterministic_evaluation(mock_env, trust_interface):
    """Test deterministic action selection during evaluation"""
    agent = DRLAgent(mock_env, 'ppo', trust_interface)
    
    # Run multiple evaluations and check if actions are consistent
    mock_env.reset.return_value = np.zeros(mock_env.observation_space.shape)
    
    actions = []
    for _ in range(3):
        obs = mock_env.reset()
        action, _ = agent.model.predict(obs, deterministic=True)
        actions.append(action)
    
    # All actions should be the same in deterministic mode
    assert all(np.array_equal(actions[0], action) for action in actions) 