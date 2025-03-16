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
        
        # Statistics
        self.episode_reward = 0.0
        self.steps = 0
        self.start_time = datetime.now()
        
        # Camera setup
        self.camera = None
        self.camera_surface = None
        self._setup_camera()

        # Simple PID control parameters
        self.prev_error = 0
        self.integral = 0
        
        # Initialize trust-based speed control
        self.base_target_speed = 20.0  # km/h at max trust
        self.min_target_speed = 0.0   # km/h at min trust
        self.target_speed = self.base_target_speed

    def run(self):
        """Main control loop"""
        try:
            self.reset()
            running = True
            
            while running:
                # Handle input and check if should continue
                running = self._handle_input()
                if not running:
                    break
                
                self.update_trust_based_speed()
                action = self.simple_pid_control()
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                self.episode_reward += reward
                self.steps += 1
                
                # Update displays
                self.display_info(info)
                self._update_plots()
                
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
            self.env.trust_interface.cleanup()

    def update_trust_based_speed(self):
        """Calculate target speed based on trust level"""
        # Linear interpolation between min and max speed based on trust
        trust_level = self.env.trust_interface.trust_level
        self.target_speed = self.min_target_speed + (self.base_target_speed - self.min_target_speed) * trust_level

    def simple_pid_control(self):  # Reduced target speed for better control
        """Simple PID control for the vehicle"""
        if not hasattr(self.env, 'vehicle'):
            return np.array([0.0, 0.0])
            
        # Calculate dt
        current_time = self.env.world.get_snapshot().timestamp.elapsed_seconds
        if hasattr(self, 'last_control_time'):
            dt = current_time - self.last_control_time
        else:
            dt = 0.1  # Default dt for first iteration
        self.last_control_time = current_time
            
        # Get current speed
        velocity = self.env.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # Convert to km/h
        
        # Get trust level and behavior adjustments if available
        trust_level = self.env.trust_interface.trust_level
        
        # Adjust PID parameters based on trust level
        # Lower trust = more conservative control (lower gains)
        trust_factor = 0.5 + 0.5 * trust_level  # Range from 0.5 to 1.0
        
        # PID parameters for speed control - adjusted by trust
        Kp_speed = 0.5 * trust_factor
        Ki_speed = 0.1 * trust_factor
        Kd_speed = 0.1 * trust_factor
        
        # Calculate error
        error = self.target_speed - speed
        
        # Anti-windup for integral term
        max_integral = 10.0
        self.integral = np.clip(self.integral + error, -max_integral, max_integral)
        
        # Calculate derivative with smoothing
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Calculate longitudinal control
        longitudinal = Kp_speed * error + Ki_speed * self.integral + Kd_speed * derivative
        
        # Convert to throttle/brake
        if longitudinal > 0:
            # Adjust throttle based on trust - lower trust means gentler acceleration
            max_throttle = 0.5 + 0.5 * trust_level  # Range from 0.5 to 1.0
            throttle = np.clip(longitudinal, 0.0, max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            # Adjust braking based on trust - lower trust means stronger braking
            brake_factor = 1.0 + (1.0 - trust_level) * 0.5  # Range from 1.0 to 1.5
            brake = np.clip(-longitudinal * brake_factor, 0.0, 1.0)
            
        # Waypoint following for steering
        try:
            # Get current waypoint and next waypoint
            current_waypoint = self.env.world.get_map().get_waypoint(self.env.vehicle.get_location())
            next_waypoint = current_waypoint.next(2.0)[0]
            
            # Get vehicle transform
            vehicle_transform = self.env.vehicle.get_transform()
            
            # Calculate vectors
            forward_vector = vehicle_transform.get_forward_vector()
            forward_vector = np.array([forward_vector.x, forward_vector.y])
            forward_vector = forward_vector / np.linalg.norm(forward_vector)
            
            target_vector = next_waypoint.transform.location - vehicle_transform.location
            target_vector = np.array([target_vector.x, target_vector.y])
            target_vector = target_vector / np.linalg.norm(target_vector)
            
            # Calculate dot product and cross product
            dot = np.clip(np.dot(forward_vector, target_vector), -1.0, 1.0)
            cross = np.cross(forward_vector, target_vector)
            
            # Calculate steering angle
            angle = np.arccos(dot)
            if cross < 0:
                angle = -angle
                
            # PID parameters for steering - adjusted by trust
            Kp_steer = 0.8 * trust_factor
            Kd_steer = 0.1 * trust_factor
            
            # Calculate steering control
            steering = Kp_steer * angle + Kd_steer * (angle / dt if dt > 0 else 0)
            
            # Adjust steering magnitude based on trust
            # Lower trust = more conservative steering (reduced magnitude)
            max_steering = 0.7 + 0.3 * trust_level  # Range from 0.7 to 1.0
            steering = np.clip(steering, -max_steering, max_steering)
            
        except:
            # Fallback to simple steering if waypoint calculation fails
            steering = 0.0
        
        # Add hesitation effect when trust is low
        # Occasionally reduce action magnitude to simulate hesitation
        if trust_level < 0.5 and np.random.random() < (1.0 - trust_level) * 0.3:
            hesitation_factor = 0.7 + 0.3 * trust_level  # Range from 0.7 to 1.0
            throttle *= hesitation_factor
            steering *= hesitation_factor
        
        # Add smoothing to prevent sudden changes
        if hasattr(self, 'prev_steering'):
            # Adjust max steering change based on trust
            # Lower trust = more gradual steering changes
            max_steering_change = 0.05 + 0.15 * trust_level  # Range from 0.05 to 0.2
            steering = np.clip(
                steering,
                self.prev_steering - max_steering_change,
                self.prev_steering + max_steering_change
            )
        self.prev_steering = steering
        
        # Debug info
        if self.steps % 20 == 0:  # Print every 20 steps
            print(f"Speed: {speed:.1f} km/h, Target: {self.target_speed:.1f} km/h")
            print(f"Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steering: {steering:.2f}")
            print(f"Trust Level: {trust_level:.2f}, Trust Factor: {trust_factor:.2f}")
        
        return np.array([steering, throttle if throttle > 0 else -brake])
    
    def reset(self):
        """Reset the environment and scenario"""
        self.env.reset()
        self.episode_reward = 0.0
        self.steps = 0
        self.start_time = datetime.now()
        self._setup_camera()  # Reset camera when environment resets

    def display_info(self, info):
        """Display current control and scenario information"""
        # Clear info surface
        self.info_surface.fill((255, 255, 255))
        
        # Create font
        font = pygame.font.Font(None, 36)
        
        # Display control info
        texts = [
            f"Trust Level: {info['trust_level']:.2f}",
            f"Target Speed: {self.target_speed:.1f} km/h",
            f"Current Speed: {self._get_current_speed():.1f} km/h",
            f"Intervention Active: {'Yes' if self.env.trust_interface.intervention_active else 'No'}",
            f"Reward: {self.episode_reward:.2f}",
            f"Steps: {self.steps}",
            f"Time: {(datetime.now() - self.start_time).seconds}s"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            self.info_surface.blit(text_surface, (10, 10 + i * 40))
        
        # Display behavior adjustments if available
        if 'behavior_adjustment' in info:
            behavior = info['behavior_adjustment']
            behavior_texts = [
                f"Behavior Factor: {behavior['behavior_factor']:.2f}",
                f"Steering Stability: {behavior['stability_factor']:.2f}",
                f"Smoothness: {behavior['smoothness_factor']:.2f}",
                f"Hesitation: {1.0 - behavior['hesitation_factor']:.2f}"
            ]
            
            for i, text in enumerate(behavior_texts):
                text_surface = font.render(text, True, (0, 0, 150))
                self.info_surface.blit(text_surface, (400, 10 + i * 40))
                
        # Display driving metrics if available
        if 'driving_metrics' in info:
            metrics = info['driving_metrics']
            y_offset = 300
            
            # Draw title
            title = font.render("Driving Metrics:", True, (0, 0, 0))
            self.info_surface.blit(title, (10, y_offset))
            
            # Draw metrics
            metric_texts = [
                f"Steering Stability: {metrics['steering_stability']:.2f}",
                f"Accel Smoothness: {metrics['acceleration_smoothness']:.2f}",
                f"Braking Smoothness: {metrics['braking_smoothness']:.2f}",
                f"Hesitation Level: {metrics['hesitation_level']:.2f}"
            ]
            
            for i, text in enumerate(metric_texts):
                text_surface = font.render(text, True, (100, 0, 0))
                self.info_surface.blit(text_surface, (10, y_offset + 40 + i * 30))
        
        # Update main display
        if self.camera_surface is not None:
            self.screen.blit(self.camera_surface, (0, 0))
        self.screen.blit(self.info_surface, (0, self.camera_height))
        
        pygame.display.flip()

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
    
    def _handle_input(self):
        """Handle keyboard input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_SPACE:
                    # Record intervention and activate emergency brake
                    self.env.trust_interface.update_trust(intervention=True, dt=0.0)
        
        return True

    def _update_plots(self):
        """Update trust history plots"""
        current_time = (datetime.now() - self.start_time).total_seconds()
        self.trust_history.append(self.env.trust_interface.trust_level)
        self.time_history.append(current_time)
        
        # Create plot
        plt.clf()
        plt.plot(list(self.time_history), list(self.trust_history), 'g-')
        plt.xlabel('Time (s)')
        plt.ylabel('Trust Level')
        plt.title('Trust Level Over Time')
        plt.grid(True)
        plt.ylim(0, 1)
        
        # Save plot
        plt.savefig('trust_plot.png')
     
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
    
    # Initialize components
    env = CarlaEnv(
        trust_interface=TrustInterface(),
        town=args.town,
        port=args.port
    )
    
    # Create and run manual controller
    controller = AutomaticController(
        env=env,
        scenario_class=scenario_map[args.scenario]
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
    
    try:
        controller.run()
    finally:
        # Stop CARLA server if we started it
        if carla_server:
            carla_server.stop_server()

if __name__ == "__main__":
    main() 