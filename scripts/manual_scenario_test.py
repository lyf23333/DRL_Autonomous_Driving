#!/usr/bin/env python

import sys
import os
import argparse
import pygame
import numpy as np
import matplotlib.pyplot as plt
import carla
from datetime import datetime
from collections import deque

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.carla_env import CarlaEnv
from src.environment.carla_env_config import CarlaEnvConfig
from src.trust.trust_interface import TrustInterface
from scenarios.lane_switching import LaneSwitchingScenario
from scenarios.urban_traffic import UrbanTrafficScenario
from scenarios.obstacle_avoidance import ObstacleAvoidanceScenario
from scenarios.emergency_braking import EmergencyBrakingScenario
from src.utils.carla_server import CarlaServerManager

class AutomaticController:
    def __init__(self, env: CarlaEnv, scenario_class):
        self.env = env
        
        # Initialize pygame for keyboard control and display
        pygame.init()
        
        # Create two windows: one for camera view and one for info
        self.camera_width = 800
        self.camera_height = 600
        self.info_width = 400
        self.info_height = 300
        
        # Main display surface (camera view)
        self.screen = pygame.display.set_mode(
            (self.camera_width, self.camera_height + self.info_height)
        )
        pygame.display.set_caption("Manual Control - Ego Vision")
        
        # Create separate surface for info display
        self.info_surface = pygame.Surface((self.camera_width, self.info_height))
        
        # Create and setup scenario
        self.scenario = scenario_class(env)
        self.env.set_scenario(self.scenario)

        # Trust plotting data
        self.trust_history = deque(maxlen=1000)
        self.time_history = deque(maxlen=1000)
        self.start_time = datetime.now()
        
        # Control parameters
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        
        # Speed limit settings
        self.max_speed = 30.0  # Maximum speed in km/h
        self.speed_limit_active = True  # Enable/disable speed limiting
        
        # Statistics
        self.episode_reward = 0.0
        self.steps = 0
        self.start_time = datetime.now()
        
        # Camera setup
        self.camera = None
        self.camera_surface = None
        self._setup_camera()


    def run(self):
        """Main control loop"""
        self.reset()
        running = True
        
        while running:
            # Handle input and check if should continue
            running = self._handle_input()
            
            # Create action array
            action = np.array([
                self.steering,
                self.throttle if self.throttle > 0 else -self.brake
            ])
            if not running:
                break
            
            # Step environment
            _, reward, done, _, info = self.env.step(action)
            self.episode_reward += reward
            self.steps += 1
            
            # Check if episode is done
            if done:
                print("\nEpisode finished!")
                print(f"Total steps: {self.steps}")
                print(f"Total reward: {self.episode_reward:.2f}")
                print(f"Average trust level: {info['trust_level']:.2f}")
                if info.get('scenario_complete', False):
                    print("Scenario completed successfully!")
                self.reset()
            
            # Small delay to make control manageable
            pygame.time.wait(50)

    def _handle_input(self):
        """Handle keyboard input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Handle key press events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                # Space for manual intervention
                elif event.key == pygame.K_SPACE:
                    self.trust_interface.record_intervention()
                # R for reset
                elif event.key == pygame.K_r:
                    self.reset()
                # L to toggle speed limit
                elif event.key == pygame.K_l:
                    self.speed_limit_active = not self.speed_limit_active
                    status = "enabled" if self.speed_limit_active else "disabled"
                    print(f"Speed limit {status} (max: {self.max_speed} km/h)")
                # Plus/Minus to adjust speed limit
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.max_speed += 5.0
                    print(f"Speed limit increased to {self.max_speed} km/h")
                elif event.key == pygame.K_MINUS:
                    self.max_speed = max(5.0, self.max_speed - 5.0)
                    print(f"Speed limit decreased to {self.max_speed} km/h")
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Steering
        if keys[pygame.K_LEFT]:
            self.steering = max(-1.0, self.steering - 0.1)
        elif keys[pygame.K_RIGHT]:
            self.steering = min(1.0, self.steering + 0.1)
        else:
            self.steering = 0.0
        
        # Get current vehicle speed
        current_speed = self._get_current_speed()
        
        # Throttle/Brake
        if keys[pygame.K_UP]:
            # Only allow throttle increase if below speed limit or if limit is disabled
            if not self.speed_limit_active or current_speed < self.max_speed:
                self.throttle = min(0.5, self.throttle + 0.1)
            else:
                # Automatically reduce throttle when speed limit is reached
                self.throttle = 0.0
            self.brake = 0.0
        elif keys[pygame.K_DOWN]:
            self.brake = min(1.0, self.brake + 0.1)
            self.throttle = 0.0
        else:
            self.throttle = 0.0
            self.brake = 0.0
        
        # Speed limit enforcement - override throttle if speed limit is active and exceeded
        if self.speed_limit_active and current_speed >= self.max_speed and self.throttle > 0:
            self.throttle = 0.0
        
        return True

    def reset(self):
        """Reset the environment and scenario"""
        self.env.reset()
        self.episode_reward = 0.0
        self.steps = 0
        self.start_time = datetime.now()
        self._setup_camera()  # Reset camera when environment resets

    """
    Helper functions
    """

    def _setup_camera(self):
        """Setup the ego vision camera"""
        if not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return
            
        # Destroy existing camera if any
        if self.camera is not None:
            self.camera.destroy()
        
        # Get the blueprint for the camera
        camera_bp = self.env.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))
        camera_bp.set_attribute('fov', '90')
        
        # Set the camera position relative to the vehicle
        camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),  # Position slightly above and forward of the hood
            carla.Rotation(pitch=-15)       # Angle slightly downward
        )
        
        # Spawn the camera
        self.camera = self.env.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.env.vehicle
        )
        
        # Create surface for camera image
        self.camera_surface = pygame.Surface((self.camera_width, self.camera_height))
        
        # Set up callback for camera data
        self.camera.listen(self._process_camera_data)
    
    def _process_camera_data(self, image):
        """Process camera data and update display"""
        # Convert CARLA raw image to pygame surface
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # Convert from BGR to RGB

        # Create a pygame surface from the array
        pygame_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        # Blit the image onto the camera surface
        self.camera_surface.blit(pygame_image, (0, 0))
     
    def _get_current_speed(self):
        """Get current vehicle speed in km/h"""
        if not hasattr(self.env, 'vehicle'):
            return 0.0
        
        velocity = self.env.vehicle.get_velocity()
        return 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # Convert to km/h

def main():
    parser = argparse.ArgumentParser(description='Manual Scenario Testing')
    parser.add_argument('--scenario', type=str, default='obstacle_avoidance',
                      choices=['lane_switching', 'urban_traffic', 'obstacle_avoidance', 'emergency_braking'],
                      help='Scenario to test')
    
    # CARLA server options
    parser.add_argument('--start-carla', action='store_true',
                      help='Start CARLA server automatically')
    parser.add_argument('--carla-path', type=str, default=None,
                      help='Path to CARLA installation (if not set, will try to auto-detect)')
    parser.add_argument('--port', type=int, default=2000,
                      help='Port to run CARLA server on')
    parser.add_argument('--town', type=str, default='Town01',
                      help='CARLA town/map to use')
    parser.add_argument('--quality', type=str, default='Epic', choices=['Low', 'Epic'],
                      help='Graphics quality for CARLA')
    parser.add_argument('--offscreen', action='store_true',
                      help='Run CARLA in offscreen mode (no rendering)')
    parser.add_argument('--max-speed', type=float, default=40.0,
                      help='Maximum speed limit in km/h')
    
    args = parser.parse_args()
    
    # Start CARLA server if requested
    carla_server = None
    if args.start_carla:
        print("Starting CARLA server...")
        carla_server = CarlaServerManager()
        success = carla_server.start_server(
            port=args.port,
            town=args.town,
            quality=args.quality,
            offscreen=args.offscreen,
            carla_path=args.carla_path
        )
        
        if not success:
            print("Failed to start CARLA server. Exiting.")
            sys.exit(1)
    
    # Scenario mapping
    scenario_map = {
        'lane_switching': LaneSwitchingScenario,
        'urban_traffic': UrbanTrafficScenario,
        'obstacle_avoidance': ObstacleAvoidanceScenario,
        'emergency_braking': EmergencyBrakingScenario
    }

    # Initialize environment with configurable parameters
    env_config = CarlaEnvConfig.from_json('configs/default_config.json')
    env_config.town = args.town
    env_config.port = args.port
    env_config.render_mode = True
    
    # Initialize components
    env = CarlaEnv(
        trust_interface=TrustInterface(),
        config=env_config
    )
    
    # Create and run manual controller
    controller = AutomaticController(
        env=env,
        scenario_class=scenario_map[args.scenario]
    )
    
    # Set the maximum speed from command line argument
    controller.max_speed = args.max_speed
    
    print("\nManual Control Instructions:")
    print("----------------------------")
    print("Arrow Keys: Control the vehicle")
    print("  ↑: Accelerate")
    print("  ↓: Brake")
    print("  ←/→: Steer")
    print("Space: Record manual intervention")
    print("L: Toggle speed limit")
    print("+/-: Increase/decrease speed limit")
    print("R: Reset episode")
    print("ESC: Quit")
    print(f"Speed limit: {controller.max_speed} km/h")
    print("----------------------------\n")
    
    try:
        controller.run()
    finally:
        # Stop CARLA server if we started it
        if carla_server:
            carla_server.stop_server()

if __name__ == "__main__":
    main() 