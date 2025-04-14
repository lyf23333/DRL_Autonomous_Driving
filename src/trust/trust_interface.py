import pygame
import numpy as np
import json
from datetime import datetime
import os
import random
import carla

class TrustInterface:
    def __init__(self, screen_width=800, screen_height=200, port = 2000):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Trust Feedback Interface")

        # Connect to CARLA server
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Trust metrics
        self.trust_level = 0.75  # Range: 0.0 to 1.0
        self.trust_increase_rate = 0.01  # Rate of trust increase during smooth operation
        self.trust_decrease_rate = 0.1  # Rate of trust decrease on intervention
        self.intervention_active = False
        
        # Speed-based intervention parameters
        self.speed_threshold_low = 10.0  # km/h - below this speed, reduced intervention probability
        self.speed_threshold_high = 30.0  # km/h - above this speed, increased intervention probability
        self.speed_factor_min = 0.01  # Minimum speed-based intervention factor
        self.speed_factor_max = 0.5  # Maximum speed-based intervention factor
        self.self_intervention_facor = 0.1
        
        # Intervention tracking
        self.manual_interventions = []
        self.intervention_timestamps = []
        self.recent_intervention_window = 5.0  # seconds to consider an intervention "recent"
        self.intervention_cooldown = 2.0  # minimum seconds between interventions
        self.last_intervention_time = 0.0

        # Enhanced intervention tracking
        self.intervention_types = {
            'brake': [],       # Timestamps of brake interventions
            'steer': [],       # Timestamps of steering corrections
            'hesitation': []   # Timestamps of hesitation events
        }
        
        # Driving behavior metrics
        self.driving_metrics = {
            'steering_stability': 1.0,  # 0.0 (unstable) to 1.0 (stable)
            'acceleration_smoothness': 1.0,  # 0.0 (jerky) to 1.0 (smooth)
            'hesitation_level': 0.0,    # 0.0 (confident) to 1.0 (hesitant)
            'speed_compliance': 1.0,    # 0.0 (excessive speeding) to 1.0 (compliant)
        }
        
        # History for calculating smoothness
        self.acceleration_history = []
        self.steering_history = []
        self.max_history_length = 20
        
        # Thresholds for detecting events
        self.steering_correction_threshold = 0.3  # Significant steering change
        self.abrupt_acceleration_threshold = 2.5  # m/s²
        self.abrupt_braking_threshold = -2.0      # m/s²
        self.hesitation_threshold = 1.5           # seconds of low speed near decision points
        
        self.current_intervention_prob = 0.0
        
        # Setup data logging
        self.data_dir = "data/trust_feedback"
        os.makedirs(self.data_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Last recorded state for calculating changes
        self.last_speed = 0.0
        self.last_steering = 0.0
        self.last_update_time = 0.0
        self.hesitation_start_time = None
        self.near_decision_point = False
        
        # Previous control state for detecting changes
        self.prev_control = None
        
        # Behavior adjustment parameters
        self.behavior_adjustment = {
            'trust_level': 0.75,
            'stability_factor': 1.0,
            'smoothness_factor': 1.0,
            'hesitation_factor': 1.0
        }
        
    def _calculate_speed_factor(self, current_speed):
        """Calculate intervention factor based on current speed"""
        if current_speed <= self.speed_threshold_low:
            return self.speed_factor_min
        elif current_speed >= self.speed_threshold_high:
            return self.speed_factor_max
        else:
            # Linear interpolation between min and max factors
            speed_range = self.speed_threshold_high - self.speed_threshold_low
            speed_ratio = (current_speed - self.speed_threshold_low) / speed_range
            return self.speed_factor_min + (self.speed_factor_max - self.speed_factor_min) * speed_ratio
        
    def update_trust(self, intervention=False, intervention_type=None, dt=0.0):
        """
        Update trust level based on interventions, driving behavior, and smooth operation
        
        Args:
            intervention: Boolean indicating if an intervention occurred
            intervention_type: Type of intervention ('brake', 'steer', 'hesitation')
            dt: Time delta since last update
        """
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds

        smoothness_factor = self.driving_metrics['acceleration_smoothness']
        stability_factor = self.driving_metrics['steering_stability']
        confidence_factor = 1.0 - self.driving_metrics['hesitation_level']
        compliance_factor = self.driving_metrics['speed_compliance']
        
        # Update trust level based on weighted driving metrics
        self.trust_level = 0.3 * smoothness_factor + 0.4 * stability_factor + 0.1 * confidence_factor + 0.2 * compliance_factor
        
        if intervention:
            self.last_intervention_time = current_time
    
    def record_intervention(self, intervention_type='brake'):
        """
        Record a manual intervention with specific type
        
        Args:
            intervention_type: Type of intervention ('brake', 'steer', 'hesitation')
        """
        self.intervention_active = True
        self.intervention_type = intervention_type
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        
        # Record in general intervention list
        self.manual_interventions.append({
            'timestamp': timestamp,
            'trust_level': self.trust_level,
            'type': intervention_type
        })
        self.intervention_timestamps.append(timestamp)
        
        # Record in type-specific list
        if intervention_type in self.intervention_types:
            self.intervention_types[intervention_type].append(timestamp)
    
    def update_driving_metrics(self, vehicle, target_speed=None):
        """
        Update driving behavior metrics based on vehicle state
        
        Args:
            vehicle: CARLA vehicle object
            target_speed: Current target speed in km/h (if provided)
        """
        if not vehicle:
            return
            
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.1
        self.last_update_time = current_time
        
        # Get current vehicle state
        velocity = vehicle.get_velocity()
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2)  # Convert to km/h (3.6 = conversion from m/s to km/h)
        current_steering = vehicle.get_control().steer
        
        # Calculate acceleration
        acceleration = (current_speed - self.last_speed) / dt if dt > 0 else 0
        
        # Update histories
        self.acceleration_history.append(acceleration)
        self.steering_history.append(current_steering)
        
        # Limit history length
        if len(self.acceleration_history) > self.max_history_length:
            self.acceleration_history.pop(0)
        if len(self.steering_history) > self.max_history_length:
            self.steering_history.pop(0)
            
        # 2. Calculate steering stability
        if len(self.steering_history) > 2:
            current_steering_change = abs(current_steering - self.steering_history[-2])
            last_steering_change = abs(self.steering_history[-2] - self.steering_history[-3])
            steering_variance = abs(current_steering_change - last_steering_change)
            steering_over_threshold = current_steering > self.steering_correction_threshold
            self.driving_metrics['steering_stability'] = max(0.0, min(1.0, 1.0 - steering_variance - steering_over_threshold * 0.2))
        
        # 3. Calculate acceleration smoothness
        if len(self.acceleration_history) > 2:
            acceleration_diff = abs(acceleration - self.acceleration_history[-2])
            acceleration_over_threshold = abs(acceleration) > self.abrupt_acceleration_threshold

            self.driving_metrics['acceleration_smoothness'] = max(0.0, min(1.0, 1.0 -acceleration_diff * 0.3 - acceleration_over_threshold * 0.2))
        
        # 4. Calculate speed compliance (new metric)
        if target_speed is not None:
            # Calculate how much the current speed exceeds the target speed
            speed_excess = max(0, current_speed * 3.6 - target_speed)
            
            # Define thresholds for speed excess
            slight_excess_threshold = 5.0    # km/h over target speed
            major_excess_threshold = 15.0    # km/h over target speed
            
            if speed_excess <= 0:
                # Speed is below or at target - maintain full compliance
                self.driving_metrics['speed_compliance'] = min(1.0, self.driving_metrics['speed_compliance'] + 0.02)
            elif speed_excess <= slight_excess_threshold:
                # Slightly over speed limit - minor penalty
                self.driving_metrics['speed_compliance'] = max(0.0, self.driving_metrics['speed_compliance'] - 0.03)
            elif speed_excess <= major_excess_threshold:
                # Moderately over speed limit - medium penalty
                self.driving_metrics['speed_compliance'] = max(0.0, self.driving_metrics['speed_compliance'] - 0.07)
            else:
                # Significantly over speed limit - major penalty
                self.driving_metrics['speed_compliance'] = max(0.0, self.driving_metrics['speed_compliance'] - 0.15)

        # 5. Detect hesitation (low speed near decision points)
        if self.near_decision_point and current_speed < 5.0 / 3.6:  # Below 5 km/h near intersection
            if self.hesitation_start_time is None:
                self.hesitation_start_time = current_time
            elif current_time - self.hesitation_start_time > self.hesitation_threshold:
                self.driving_metrics['hesitation_level'] = min(1.0, self.driving_metrics['hesitation_level'] + 0.1)
                self.hesitation_start_time = None  # Reset after recording
        else:
            self.hesitation_start_time = None
            # Gradually decrease hesitation level
            self.driving_metrics['hesitation_level'] = max(0.0, self.driving_metrics['hesitation_level'] - 0.01)
        
        # Update last recorded state
        self.last_speed = current_speed
        self.last_steering = current_steering
    
    def set_near_decision_point(self, is_near):
        """
        Set whether the vehicle is near a decision point (intersection, lane merge, etc.)
        
        Args:
            is_near: Boolean indicating if vehicle is near a decision point
        """
        self.near_decision_point = is_near

    def get_recent_interventions(self):
        """Get number of recent interventions within the window"""
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        recent_count = sum(1 for t in self.intervention_timestamps 
                         if current_time - t <= self.recent_intervention_window)
        return recent_count
    
    def get_recent_interventions_by_type(self, intervention_type):
        """
        Get number of recent interventions of a specific type
        
        Args:
            intervention_type: Type of intervention to count
        """
        if intervention_type not in self.intervention_types:
            return 0
            
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        recent_count = sum(1 for t in self.intervention_types[intervention_type] 
                         if current_time - t <= self.recent_intervention_window)
        return recent_count
    
    def get_intervention_observation(self):
        """Get binary observation of recent intervention"""
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        return int(any(current_time - t <= self.recent_intervention_window 
                      for t in self.intervention_timestamps))
    
    def get_current_trust_state(self):
        """Get the current trust state and driving metrics"""
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        
        # Count recent interventions by type
        recent_interventions = {
            'total': self.get_recent_interventions(),
            'brake': self.get_recent_interventions_by_type('brake'),
            'steer': self.get_recent_interventions_by_type('steer'),
            'hesitation': self.get_recent_interventions_by_type('hesitation')
        }
        
        return {
            'trust_level': self.trust_level,
            'recent_interventions': recent_interventions,
            'time_since_last_intervention': current_time - self.last_intervention_time 
                if self.intervention_timestamps else float('inf'),
            'driving_metrics': self.driving_metrics
        }
    
    def save_session_data(self):
        """Save the trust feedback data for the session"""
        data = {
            'session_id': self.session_id,
            'manual_interventions': self.manual_interventions,
            'final_trust_level': self.trust_level,
            'intervention_count': len(self.manual_interventions),
            'intervention_types': {
                'brake': len(self.intervention_types['brake']),
                'steer': len(self.intervention_types['steer']),
                'hesitation': len(self.intervention_types['hesitation'])
            },
            'final_driving_metrics': self.driving_metrics
        }
        
        filename = os.path.join(self.data_dir, f"trust_data_{self.session_id}.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    def cleanup(self):
        """Clean up pygame resources"""
        self.save_session_data()
        pygame.quit()


    def detect_interventions_and_update_trust(self, current_control, prev_control, world_snapshot=None, dt=0.0):
        """
        Detect manual interventions based on control changes and update trust accordingly.
        
        Args:
            current_control: Current vehicle control
            prev_control: Previous vehicle control
            world_snapshot: CARLA world snapshot for timestamp
            dt: Time delta since last update
            
        Returns:
            intervention_detected: Whether an intervention was detected
        """
        if prev_control is None:
            return False
            
        # Check if an intervention is already active
        if self.intervention_active:
            # An intervention was already recorded
            self.update_trust(intervention=True, intervention_type=self.intervention_type, dt=dt)
            
            # Reset intervention flag after processing
            self.intervention_active = False
            self.intervention_type = None
            return True
        else:
            self.update_trust(intervention=False, dt=dt)
            return False 

    def reset(self):
        """Reset the trust interface"""
        # Reset trust level to initial value
        self.trust_level = 0.75
        
        # Reset intervention flags
        self.intervention_active = False
        
        # Clear intervention history
        self.manual_interventions = []
        self.intervention_timestamps = []
        self.last_intervention_time = 0.0
        
        # Clear intervention type history
        for key in self.intervention_types:
            self.intervention_types[key] = []
        
        # Reset driving metrics
        self.driving_metrics = {
            'steering_stability': 1.0,
            'acceleration_smoothness': 1.0,
            'hesitation_level': 0.0,
            'speed_compliance': 1.0,
        }
        
        # Reset behavior adjustment
        self.behavior_adjustment = {
            'trust_level': 0.75,
            'stability_factor': 1.0,
            'smoothness_factor': 1.0,
            'hesitation_factor': 1.0
        }
        
        # Reset history buffers
        self.acceleration_history = []
        self.steering_history = []
        
        # Reset last recorded state
        self.last_speed = 0.0
        self.last_steering = 0.0
        self.last_update_time = 0.0
        self.hesitation_start_time = None
        self.near_decision_point = False
        
        # Reset episode intervention counter
        self.episode_intervention_count = 0

    def update_behavior_adjustment(self):
        """Update behavior adjustment parameters based on trust level and driving metrics"""
        # Get current trust level
        trust_level = self.trust_level
        
        # Get relevant metrics
        stability_factor = self.driving_metrics['steering_stability']
        smoothness_factor = self.driving_metrics['acceleration_smoothness']
        hesitation_factor = 1.0 - self.driving_metrics['hesitation_level']
        
        # Store these for potential use in action modification
        self.behavior_adjustment = {
            'trust_level': trust_level,
            'stability_factor': stability_factor,
            'smoothness_factor': smoothness_factor,
            'hesitation_factor': hesitation_factor
        }
        
        return self.behavior_adjustment
        
    def get_behavior_adjustment(self):
        """Get the current behavior adjustment parameters"""
        return self.behavior_adjustment 

    def update_display(self):
        """Update the trust feedback display with enhanced metrics"""
        self.screen.fill(self.WHITE)
        
        # Draw trust level bar
        bar_width = 600
        bar_height = 30
        x = 100
        y = 30
        
        # Background bar
        pygame.draw.rect(self.screen, self.BLACK, 
                        (x, y, bar_width, bar_height), 2)
        
        # Trust level fill
        fill_width = int(bar_width * self.trust_level)
        pygame.draw.rect(self.screen, self.GREEN,
                        (x, y, fill_width, bar_height))
        
        # Draw text
        font = pygame.font.Font(None, 28)
        trust_text = f"Trust Level: {self.trust_level:.2f}"
        text_surface = font.render(trust_text, True, self.BLACK)
        self.screen.blit(text_surface, (x, y - 25))
        
        # Draw intervention counts
        y += 40
        interventions = self.get_current_trust_state()['recent_interventions']
        
        intervention_texts = [
            f"Recent Interventions: {interventions['total']}",
            f"Braking: {interventions['brake']}",
            f"Steering: {interventions['steer']}",
            f"Hesitations: {interventions['hesitation']}"
        ]
        
        for i, text in enumerate(intervention_texts):
            text_surface = font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (x, y + i * 25))
        
        # Draw driving metrics
        y += 150
        metrics = self.driving_metrics
        
        # Draw metric bars
        metric_names = [
            "Steering Stability", 
            "Acceleration Smoothness", 
            "Confidence (vs Hesitation)",
            "Speed Compliance"
        ]
        
        metric_values = [
            metrics['steering_stability'],
            metrics['acceleration_smoothness'],
            1.0 - metrics['hesitation_level'],
            metrics['speed_compliance']
        ]
        
        metric_colors = [
            self.BLUE,
            self.GREEN,
            self.YELLOW,
            self.GREEN,
        ]
        
        for i, (name, value, color) in enumerate(zip(metric_names, metric_values, metric_colors)):
            # Draw label
            text_surface = font.render(name, True, self.BLACK)
            self.screen.blit(text_surface, (x, y + i * 25))
            
            # Draw background bar
            pygame.draw.rect(self.screen, self.BLACK, 
                            (x + 200, y + i * 25, 200, 20), 1)
            
            # Draw value bar
            fill_width = int(200 * value)
            pygame.draw.rect(self.screen, color,
                            (x + 200, y + i * 25, fill_width, 20))
            
            # Draw value text
            value_text = f"{value:.2f}"
            text_surface = font.render(value_text, True, self.BLACK)
            self.screen.blit(text_surface, (x + 410, y + i * 25))
        
        pygame.display.flip()

    def get_intervention_count(self):
        """Get the total number of interventions in the current episode"""
        if not hasattr(self, 'episode_intervention_count'):
            self.episode_intervention_count = 0
        return self.episode_intervention_count
        
    def record_intervention(self, intervention_type='brake'):
        """Record an intervention of a specific type
        
        Args:
            intervention_type: Type of intervention ('brake', 'steer', or 'hesitation')
        """
        # Add to specific intervention type history
        if intervention_type in self.intervention_types:
            current_time = self.world.get_snapshot().timestamp.elapsed_seconds
            self.intervention_types[intervention_type].append(current_time)
        
        # Add to general intervention history
        self.manual_interventions.append(1.0)  # Placeholder for intervention magnitude
        self.intervention_timestamps.append(self.world.get_snapshot().timestamp.elapsed_seconds)
        
        # Track intervention count for the episode
        if not hasattr(self, 'episode_intervention_count'):
            self.episode_intervention_count = 0
        self.episode_intervention_count += 1