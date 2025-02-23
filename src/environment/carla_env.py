import carla
import gym
import numpy as np
from gym import spaces

class CarlaEnv(gym.Env):
    """Custom Carla environment that follows gym interface"""
    
    def __init__(self, town='Town01', port=2000):
        super(CarlaEnv, self).__init__()
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Scenario management
        self.active_scenario = None
        self.scenario_config = None
        
        # Trust-related attributes
        self.trust_interface = None
        self.last_step_time = None
        self.intervention_active = False
        
        # Set up action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # [steering, throttle/brake]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space includes vehicle state and intervention history
        # but NOT trust level (agent shouldn't observe trust directly)
        self.observation_space = spaces.Dict({
            'vehicle_state': spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),  # [speed_x, speed_y, accel_x, accel_y]
                high=np.array([np.inf, np.inf, np.inf, np.inf]),
                dtype=np.float32
            ),
            'recent_intervention': spaces.Discrete(2),  # Binary: 0 or 1
            'scenario_obs': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20,),  # Adjust size based on scenario needs
                dtype=np.float32
            )
        })
        
    def set_scenario(self, scenario, config=None):
        """Set the active scenario for the environment"""
        self.active_scenario = scenario
        self.scenario_config = config
        
    def set_trust_interface(self, trust_interface):
        """Set the trust interface for the environment"""
        self.trust_interface = trust_interface
        
    def reset(self):
        """Reset the environment to initial state"""
        # Reset the simulation
        self.world.tick()
        
        # Reset trust-related variables
        self.last_step_time = None
        self.intervention_active = False
        
        # Spawn ego vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        
        if len(spawn_points) > 0:
            spawn_point = spawn_points[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Setup active scenario if exists
        if self.active_scenario:
            self.active_scenario.setup()
        
        return self._get_obs()
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Get current time for trust updates
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        
        # Calculate dt for trust updates
        if self.last_step_time is not None:
            dt = current_time - self.last_step_time
        else:
            dt = 0.0
        self.last_step_time = current_time
        
        # Check for trust-based intervention
        if self.trust_interface is not None:
            should_intervene = self.trust_interface.should_intervene(current_time)
            if should_intervene:
                # Override action with emergency brake
                action = np.array([0.0, -1.0])  # No steering, full brake
                self.intervention_active = True
            else:
                self.intervention_active = False
            
            # Update trust level based on intervention and time delta
            self.trust_interface.update_trust(self.intervention_active, dt)
        
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
            'trust_level': self.trust_interface.trust_level if self.trust_interface else 0.5,
            'intervention_active': self.intervention_active,
            'recent_interventions': self.trust_interface.get_recent_interventions() if self.trust_interface else 0
        }
        
        # Add scenario-specific info
        if self.active_scenario:
            info['scenario_complete'] = self.active_scenario.check_scenario_completion()
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get current observation of the environment"""
        if self.vehicle is None:
            return {
                'vehicle_state': np.zeros(4),
                'recent_intervention': 0,
                'scenario_obs': np.zeros(20)
            }
        
        # Get vehicle state
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        
        vehicle_state = np.array([
            velocity.x, velocity.y,
            acceleration.x, acceleration.y
        ])
        
        # Get intervention state (binary indicator of recent intervention)
        recent_intervention = (
            self.trust_interface.get_intervention_observation()
            if self.trust_interface is not None else 0
        )
        
        # Get scenario-specific observations
        if self.active_scenario:
            scenario_obs = self.active_scenario.get_scenario_specific_obs()
        else:
            scenario_obs = np.zeros(20)
        
        # Ensure scenario_obs has correct size
        scenario_obs = np.pad(
            scenario_obs,
            (0, 20 - len(scenario_obs)),
            'constant',
            constant_values=0
        )
        
        return {
            'vehicle_state': vehicle_state,
            'recent_intervention': recent_intervention,
            'scenario_obs': scenario_obs
        }
    
    def _calculate_reward(self):
        """Calculate reward based on current state"""
        if not self.active_scenario:
            return 0.0
            
        # Basic reward components
        progress_reward = 0.0  # Based on distance to goal
        safety_reward = 0.0    # Based on distance to obstacles/vehicles
        trust_reward = self.trust_interface.trust_level if self.trust_interface else 0.5
        
        # Penalty for interventions
        intervention_penalty = -1.0 * self.intervention_active
        
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
        if self.intervention_active:
            return True
            
        if self.active_scenario and self.active_scenario.check_scenario_completion():
            return True
            
        return False
    
    def close(self):
        """Cleanup the environment"""
        if self.active_scenario:
            self.active_scenario.cleanup()
        
        if hasattr(self, 'vehicle'):
            self.vehicle.destroy() 