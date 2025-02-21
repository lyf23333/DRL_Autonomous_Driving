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
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # Adjust based on actual observation vector size
            dtype=np.float32
        )
        
        # Trust-related attributes
        self.trust_level = 0.5  # Initialize with neutral trust
        self.manual_interventions = 0
        self.intervention_threshold = 5
        
    def reset(self):
        """Reset the environment to initial state"""
        # Reset the simulation
        self.world.tick()
        
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        
        if len(spawn_points) > 0:
            spawn_point = spawn_points[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Reset trust metrics
        self.trust_level = 0.5
        self.manual_interventions = 0
        
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
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward based on:
        # - Progress towards goal
        # - Safety metrics
        # - Trust level
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'trust_level': self.trust_level,
            'manual_interventions': self.manual_interventions
        }
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get current observation of the environment"""
        if self.vehicle is None:
            return np.zeros(self.observation_space.shape)
        
        # Get vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        # Combine all observations
        obs = np.array([
            transform.location.x,
            transform.location.y,
            transform.rotation.yaw,
            velocity.x,
            velocity.y,
            self.trust_level,
            # Add more observations as needed
        ])
        
        return obs
    
    def _calculate_reward(self):
        """Calculate reward based on current state"""
        # Basic reward structure
        reward = 0
        
        # Add reward components based on:
        # 1. Progress towards goal
        # 2. Safety (distance to obstacles)
        # 3. Trust level
        # 4. Smooth driving
        
        return reward
    
    def _is_done(self):
        """Check if episode is done"""
        # Episode ends if:
        # 1. Goal is reached
        # 2. Collision occurs
        # 3. Too many manual interventions
        # 4. Time limit exceeded
        if self.manual_interventions > self.intervention_threshold:
            return True
            
        return False
    
    def update_trust(self, intervention):
        """Update trust level based on manual interventions"""
        if intervention:
            self.manual_interventions += 1
            self.trust_level = max(0.0, self.trust_level - 0.1)
        else:
            self.trust_level = min(1.0, self.trust_level + 0.05) 