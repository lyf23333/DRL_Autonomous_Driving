import carla
import gym
import numpy as np
from gym import spaces
from .trust_interface import TrustInterface


class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gym interface"""
    
    def __init__(self, town='Town01', port=2000, trust_interface: TrustInterface | None = None):
        self._initialized = False
        super(CarlaEnv, self).__init__()
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Scenario management
        self.active_scenario = None
        self.scenario_config = None
        
        # Set up action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # [steering, throttle/brake]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space will include:
        # - Vehicle state (position, velocity, acceleration)
        # - Sensor data (cameras, lidar)
        # - Trust metrics
        # - Scenario-specific observations
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),  # Increased to accommodate scenario observations
            dtype=np.float32
        )
        
        # Trust-related attributes
        self.trust_level = 0.5  # Initialize with neutral trust
        self.manual_interventions = 0
        self.intervention_threshold = 5
        
    def set_scenario(self, scenario, config=None):
        """Set the active scenario for the environment"""
        self.active_scenario = scenario
        self.scenario_config = config
        
    def reset(self):
        """Reset the environment to initial state"""
        # Reset the simulation
        self.world.tick()
        
        # Spawn ego vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        
        if len(spawn_points) > 0:
            spawn_point = spawn_points[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Reset trust metrics
        self.trust_level = 0.5
        self.manual_interventions = 0
        
        # Setup active scenario if exists
        if self.active_scenario:
            self.active_scenario.setup()
        
        return self._get_obs()
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Apply action
        control = carla.VehicleControl(
            throttle=float(action[1]) if action[1] > 0 else 0,
            brake=float(-action[1]) if action[1] < 0 else 0,
            steer=float(action[0])
        )
        self.vehicle.apply_control(control)
        
        # Tick the simulation
        self.world.tick()
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'trust_level': self.trust_level,
            'manual_interventions': self.manual_interventions
        }
        
        # Add scenario-specific info if available
        if self.active_scenario:
            info['scenario_complete'] = self.active_scenario.check_scenario_completion()
        
        if self.trust_interface:
            self.trust_interface.update_display()
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get current observation of the environment"""
        if self.vehicle is None:
            return np.zeros(self.observation_space.shape)
        
        # Get vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        # Basic observations
        basic_obs = np.array([
            transform.location.x,
            transform.location.y,
            transform.rotation.yaw,
            velocity.x,
            velocity.y,
            self.trust_level
        ])
        
        # Get scenario-specific observations
        if self.active_scenario:
            scenario_obs = self.active_scenario.get_scenario_specific_obs()
        else:
            scenario_obs = np.zeros(5)  # Default size for scenario observations
        
        # Combine all observations
        return np.concatenate([basic_obs, scenario_obs])
    
    def _calculate_reward(self):
        """Calculate reward based on current state"""
        if not self.active_scenario:
            return 0.0
            
        # Basic reward components
        progress_reward = 0.0  # Based on distance to goal
        safety_reward = 0.0    # Based on distance to obstacles/vehicles
        trust_reward = self.trust_level  # Higher trust = higher reward
        
        # Penalty for interventions
        intervention_penalty = -1.0 * self.manual_interventions
        
        # Check scenario completion
        if self.active_scenario.check_scenario_completion():
            completion_reward = 10.0
        else:
            completion_reward = 0.0
        
        # Combine rewards
        total_reward = (
            progress_reward +
            safety_reward +
            trust_reward +
            intervention_penalty +
            completion_reward
        )
        
        return total_reward
    
    def _is_done(self):
        """Check if episode is done"""
        if self.manual_interventions > self.intervention_threshold:
            return True
            
        if self.active_scenario and self.active_scenario.check_scenario_completion():
            return True
            
        return False
    
    def update_trust(self, intervention):
        """Update trust level based on manual interventions"""
        if intervention:
            self.manual_interventions += 1
            self.trust_level = max(0.0, self.trust_level - 0.1)
        else:
            self.trust_level = min(1.0, self.trust_level + 0.05)
    
    def close(self):
        """Cleanup the environment"""
        if self.active_scenario:
            self.active_scenario.cleanup()
        
        if hasattr(self, 'vehicle'):
            self.vehicle.destroy() 

        if self.trust_interface:
            self.trust_interface.cleanup()