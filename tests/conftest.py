import pytest
import carla
import os
import sys
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.carla_env import CarlaEnv
from src.trust.trust_interface import TrustInterface
from src.agents.drl_agent import DRLAgent

@pytest.fixture
def mock_carla_world():
    """Create a mock CARLA world"""
    mock_world = MagicMock()
    
    # Mock blueprint library
    mock_bp_lib = MagicMock()
    mock_bp_lib.find.return_value = MagicMock()
    mock_bp_lib.filter.return_value = [MagicMock()]
    mock_world.get_blueprint_library.return_value = mock_bp_lib
    
    # Mock spawn points
    mock_spawn_point = carla.Transform()
    mock_world.get_map.return_value.get_spawn_points.return_value = [mock_spawn_point]
    
    # Mock actor spawning
    mock_vehicle = MagicMock()
    mock_vehicle.get_location.return_value = carla.Location(x=0, y=0, z=0)
    mock_vehicle.get_transform.return_value = carla.Transform()
    mock_vehicle.get_velocity.return_value = carla.Vector3D(0, 0, 0)
    mock_world.spawn_actor.return_value = mock_vehicle
    
    return mock_world

@pytest.fixture
def mock_carla_client(mock_carla_world):
    """Create a mock CARLA client"""
    mock_client = MagicMock()
    mock_client.get_world.return_value = mock_carla_world
    return mock_client

@pytest.fixture
def mock_env(mock_carla_client):
    """Create a mock environment"""
    env = CarlaEnv()
    env.client = mock_carla_client
    env.world = mock_carla_client.get_world()
    return env

@pytest.fixture
def trust_interface():
    """Create a trust interface instance"""
    return TrustInterface(headless=True)  # Run in headless mode for testing

@pytest.fixture
def drl_agent(mock_env, trust_interface):
    """Create a DRL agent instance"""
    return DRLAgent(
        env=mock_env,
        algorithm='ppo',
        trust_interface=trust_interface
    ) 