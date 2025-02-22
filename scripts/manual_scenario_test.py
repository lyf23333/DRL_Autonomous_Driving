#!/usr/bin/env python

import sys
import os
import argparse
import pygame
import numpy as np
import carla
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.carla_env import CarlaEnv
from src.trust.trust_interface import TrustInterface
from scenarios.lane_switching import LaneSwitchingScenario
from scenarios.urban_traffic import UrbanTrafficScenario
from scenarios.obstacle_avoidance import ObstacleAvoidanceScenario

class ManualController:
    def __init__(self, env: CarlaEnv, scenario_class, trust_interface: TrustInterface):
        self.env = env
        self.trust_interface = trust_interface
        
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
        
        # Control parameters
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        
        # Statistics
        self.episode_reward = 0.0
        self.steps = 0
        self.start_time = datetime.now()
        
        # Camera setup
        self.camera = None
        self.camera_surface = None
        self.setup_camera()
        
    def setup_camera(self):
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
        self.camera.listen(self.process_camera_data)
    
    def process_camera_data(self, image):
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

    
    def handle_input(self):
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
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Steering
        if keys[pygame.K_LEFT]:
            self.steering = min(1.0, self.steering + 0.1)
        elif keys[pygame.K_RIGHT]:
            self.steering = max(-1.0, self.steering - 0.1)
        else:
            self.steering = 0.0
        
        # Throttle/Brake
        if keys[pygame.K_UP]:
            self.throttle = min(1.0, self.throttle + 0.1)
            self.brake = 0.0
        elif keys[pygame.K_DOWN]:
            self.brake = min(1.0, self.brake + 0.1)
            self.throttle = 0.0
        else:
            self.throttle = 0.0
            self.brake = 0.0
        
        return True
    
    def reset(self):
        """Reset the environment and scenario"""
        self.env.reset()
        self.episode_reward = 0.0
        self.steps = 0
        self.start_time = datetime.now()
        self.setup_camera()  # Reset camera when environment resets
    
    def run(self):
        """Main control loop"""
        try:
            self.reset()
            running = True
            
            while running:
                # Handle input
                running = self.handle_input()
                
                # Create action array
                action = np.array([
                    self.steering,
                    self.throttle if self.throttle > 0 else -self.brake
                ])
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                self.episode_reward += reward
                self.steps += 1
                
                # Update displays
                # self.trust_interface.update_display()
                self.display_info(info)
                
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
        
        finally:
            if self.camera is not None:
                self.camera.destroy()
            pygame.quit()
            self.env.close()
            self.trust_interface.cleanup()
    
    def display_info(self, info):
        """Display current control and scenario information"""
        # Clear info surface
        self.info_surface.fill((255, 255, 255))
        
        # Create font
        font = pygame.font.Font(None, 36)
        
        # Display control info
        texts = [
            f"Steering: {self.steering:.2f}",
            f"Throttle: {self.throttle:.2f}",
            f"Brake: {self.brake:.2f}",
            f"Trust Level: {info['trust_level']:.2f}",
            f"Reward: {self.episode_reward:.2f}",
            f"Steps: {self.steps}",
            f"Time: {(datetime.now() - self.start_time).seconds}s"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            self.info_surface.blit(text_surface, (10, 10 + i * 40))
        
        # Update main display
        if self.camera_surface is not None:
            self.screen.blit(self.camera_surface, (0, 0))
        self.screen.blit(self.info_surface, (0, self.camera_height))
        
        pygame.display.flip()

def main():
    parser = argparse.ArgumentParser(description='Manual Scenario Testing')
    parser.add_argument('--scenario', type=str, default='obstacle_avoidance',
                      choices=['lane_switching', 'urban_traffic', 'obstacle_avoidance'],
                      help='Scenario to test')
    args = parser.parse_args()
    
    # Scenario mapping
    scenario_map = {
        'lane_switching': LaneSwitchingScenario,
        'urban_traffic': UrbanTrafficScenario,
        'obstacle_avoidance': ObstacleAvoidanceScenario
    }
    
    # Initialize components
    env = CarlaEnv()
    trust_interface = TrustInterface()
    
    # Create and run manual controller
    controller = ManualController(
        env=env,
        scenario_class=scenario_map[args.scenario],
        trust_interface=trust_interface
    )
    
    print("\nManual Control Instructions:")
    print("----------------------------")
    print("Arrow Keys: Control the vehicle")
    print("  ↑: Accelerate")
    print("  ↓: Brake")
    print("  ←/→: Steer")
    print("Space: Record manual intervention")
    print("R: Reset episode")
    print("ESC: Quit")
    print("----------------------------\n")
    
    controller.run()

if __name__ == '__main__':
    main() 