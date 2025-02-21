import pytest
import numpy as np
from scenarios.lane_switching import LaneSwitchingScenario
from scenarios.urban_traffic import UrbanTrafficScenario
from scenarios.obstacle_avoidance import ObstacleAvoidanceScenario

@pytest.fixture
def lane_switching_scenario(mock_env):
    return LaneSwitchingScenario(mock_env)

@pytest.fixture
def urban_traffic_scenario(mock_env):
    return UrbanTrafficScenario(mock_env)

@pytest.fixture
def obstacle_avoidance_scenario(mock_env):
    return ObstacleAvoidanceScenario(mock_env)

class TestLaneSwitchingScenario:
    def test_initialization(self, lane_switching_scenario):
        """Test scenario initialization"""
        assert lane_switching_scenario.env is not None
        assert lane_switching_scenario.world is not None
    
    def test_setup(self, lane_switching_scenario):
        """Test scenario setup"""
        lane_switching_scenario.setup()
        assert hasattr(lane_switching_scenario, 'other_vehicle')
    
    def test_scenario_specific_obs(self, lane_switching_scenario):
        """Test scenario-specific observations"""
        lane_switching_scenario.setup()
        obs = lane_switching_scenario.get_scenario_specific_obs()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)  # [x, y, yaw, vel_x, vel_y]
    
    def test_cleanup(self, lane_switching_scenario):
        """Test scenario cleanup"""
        lane_switching_scenario.setup()
        lane_switching_scenario.cleanup()
        assert not hasattr(lane_switching_scenario, 'other_vehicle')

class TestUrbanTrafficScenario:
    def test_initialization(self, urban_traffic_scenario):
        """Test scenario initialization"""
        assert urban_traffic_scenario.env is not None
        assert urban_traffic_scenario.world is not None
        assert len(urban_traffic_scenario.vehicles) == 0
        assert len(urban_traffic_scenario.walkers) == 0
    
    def test_setup(self, urban_traffic_scenario):
        """Test scenario setup"""
        urban_traffic_scenario.setup()
        assert len(urban_traffic_scenario.vehicles) > 0
        assert len(urban_traffic_scenario.walkers) > 0
    
    def test_traffic_vehicle_spawning(self, urban_traffic_scenario):
        """Test traffic vehicle spawning"""
        urban_traffic_scenario._spawn_traffic_vehicles(num_vehicles=5)
        assert len(urban_traffic_scenario.vehicles) > 0
        for vehicle in urban_traffic_scenario.vehicles:
            assert vehicle is not None
    
    def test_pedestrian_spawning(self, urban_traffic_scenario):
        """Test pedestrian spawning"""
        urban_traffic_scenario._spawn_pedestrians(num_pedestrians=3)
        assert len(urban_traffic_scenario.walkers) > 0
        for walker in urban_traffic_scenario.walkers:
            assert walker is not None
    
    def test_cleanup(self, urban_traffic_scenario):
        """Test scenario cleanup"""
        urban_traffic_scenario.setup()
        urban_traffic_scenario.cleanup()
        assert len(urban_traffic_scenario.vehicles) == 0
        assert len(urban_traffic_scenario.walkers) == 0

class TestObstacleAvoidanceScenario:
    def test_initialization(self, obstacle_avoidance_scenario):
        """Test scenario initialization"""
        assert obstacle_avoidance_scenario.env is not None
        assert obstacle_avoidance_scenario.world is not None
        assert len(obstacle_avoidance_scenario.obstacles) == 0
        assert len(obstacle_avoidance_scenario.obstacle_sensors) == 0
    
    def test_setup(self, obstacle_avoidance_scenario):
        """Test scenario setup"""
        obstacle_avoidance_scenario.setup()
        assert len(obstacle_avoidance_scenario.obstacles) > 0
        assert len(obstacle_avoidance_scenario.obstacle_sensors) > 0
    
    def test_static_obstacle_spawning(self, obstacle_avoidance_scenario):
        """Test static obstacle spawning"""
        obstacle_avoidance_scenario._spawn_static_obstacles(num_obstacles=3)
        assert len(obstacle_avoidance_scenario.obstacles) > 0
        for obstacle in obstacle_avoidance_scenario.obstacles:
            assert obstacle is not None
    
    def test_dynamic_obstacle_spawning(self, obstacle_avoidance_scenario):
        """Test dynamic obstacle spawning"""
        obstacle_avoidance_scenario._spawn_dynamic_obstacles(num_obstacles=2)
        assert len(obstacle_avoidance_scenario.obstacles) > 0
        for obstacle in obstacle_avoidance_scenario.obstacles:
            assert obstacle is not None
    
    def test_sensor_setup(self, obstacle_avoidance_scenario):
        """Test sensor setup"""
        obstacle_avoidance_scenario._setup_sensors()
        assert len(obstacle_avoidance_scenario.obstacle_sensors) > 0
        for sensor in obstacle_avoidance_scenario.obstacle_sensors:
            assert sensor is not None
    
    def test_cleanup(self, obstacle_avoidance_scenario):
        """Test scenario cleanup"""
        obstacle_avoidance_scenario.setup()
        obstacle_avoidance_scenario.cleanup()
        assert len(obstacle_avoidance_scenario.obstacles) == 0
        assert len(obstacle_avoidance_scenario.obstacle_sensors) == 0 