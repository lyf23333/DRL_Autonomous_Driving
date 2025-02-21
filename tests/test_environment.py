import pytest
import numpy as np
from src.environment.carla_env import CarlaEnv

def test_env_initialization(mock_env):
    """Test environment initialization"""
    assert mock_env.trust_level == 0.5
    assert mock_env.manual_interventions == 0
    assert isinstance(mock_env.action_space.sample(), np.ndarray)
    assert isinstance(mock_env.observation_space.sample(), np.ndarray)

def test_env_reset(mock_env):
    """Test environment reset"""
    obs = mock_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == mock_env.observation_space.shape
    assert mock_env.trust_level == 0.5
    assert mock_env.manual_interventions == 0

def test_env_step(mock_env):
    """Test environment step"""
    mock_env.reset()
    action = mock_env.action_space.sample()
    
    obs, reward, done, info = mock_env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'trust_level' in info
    assert 'manual_interventions' in info

def test_trust_update(mock_env):
    """Test trust level updates"""
    initial_trust = mock_env.trust_level
    
    # Test trust decrease on intervention
    mock_env.update_trust(intervention=True)
    assert mock_env.trust_level < initial_trust
    assert mock_env.manual_interventions == 1
    
    # Test trust increase on good behavior
    current_trust = mock_env.trust_level
    mock_env.update_trust(intervention=False)
    assert mock_env.trust_level > current_trust
    assert mock_env.trust_level <= 1.0

def test_observation_space(mock_env):
    """Test observation space properties"""
    obs = mock_env.reset()
    
    assert obs.shape == mock_env.observation_space.shape
    assert np.all(obs >= mock_env.observation_space.low)
    assert np.all(obs <= mock_env.observation_space.high)

def test_action_space(mock_env):
    """Test action space properties"""
    action = mock_env.action_space.sample()
    
    assert action.shape == (2,)  # [steering, throttle/brake]
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

def test_reward_calculation(mock_env):
    """Test reward calculation"""
    mock_env.reset()
    action = np.array([0.0, 0.5])  # Moderate acceleration, no steering
    
    _, reward, _, _ = mock_env.step(action)
    assert isinstance(reward, (int, float))
    
    # Test collision penalty
    mock_env.vehicle.get_location.return_value.distance.return_value = 0.1  # Simulate collision
    _, reward_collision, _, _ = mock_env.step(action)
    assert reward_collision < reward

def test_done_conditions(mock_env):
    """Test episode termination conditions"""
    mock_env.reset()
    
    # Test termination on too many interventions
    for _ in range(mock_env.intervention_threshold + 1):
        mock_env.update_trust(intervention=True)
    
    _, _, done, _ = mock_env.step(mock_env.action_space.sample())
    assert done

def test_env_cleanup(mock_env):
    """Test environment cleanup"""
    mock_env.reset()
    mock_env.close()
    
    # Verify that the vehicle was destroyed
    if hasattr(mock_env, 'vehicle'):
        mock_env.vehicle.destroy.assert_called_once() 