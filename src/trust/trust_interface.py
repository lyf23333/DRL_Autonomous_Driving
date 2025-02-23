import pygame
import numpy as np
import json
from datetime import datetime
import os
import random

class TrustInterface:
    def __init__(self, screen_width=800, screen_height=200):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Trust Feedback Interface")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        
        # Trust metrics
        self.trust_level = 0.5  # Range: 0.0 to 1.0
        self.trust_increase_rate = 0.01  # Rate of trust increase during smooth operation
        self.trust_decrease_rate = 0.05  # Rate of trust decrease on intervention
        
        # Intervention tracking
        self.manual_interventions = []
        self.intervention_timestamps = []
        self.recent_intervention_window = 5.0  # seconds to consider an intervention "recent"
        self.intervention_cooldown = 2.0  # minimum seconds between interventions
        self.last_intervention_time = 0.0
        
        # Setup data logging
        self.data_dir = "data/trust_feedback"
        os.makedirs(self.data_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def should_intervene(self, current_time):
        """Determine if an intervention should occur based on trust level"""
        # Check cooldown
        if current_time - self.last_intervention_time < self.intervention_cooldown:
            return False
            
        # Probability of intervention increases as trust decreases
        return random.random() < (1 - self.trust_level)
    
    def update_trust(self, intervention, dt):
        """Update trust level based on interventions and smooth operation"""
        if intervention:
            # Decrease trust on intervention
            self.trust_level = max(0.0, self.trust_level - self.trust_decrease_rate)
            self.last_intervention_time = datetime.now().timestamp()
            self.record_intervention()
        else:
            # Gradually increase trust during smooth operation
            self.trust_level = min(1.0, self.trust_level + self.trust_increase_rate * dt)
    
    def record_intervention(self):
        """Record a manual intervention"""
        timestamp = datetime.now().timestamp()
        self.manual_interventions.append({
            'timestamp': timestamp,
            'trust_level': self.trust_level
        })
        self.intervention_timestamps.append(timestamp)
    
    def get_recent_interventions(self):
        """Get number of recent interventions within the window"""
        current_time = datetime.now().timestamp()
        recent_count = sum(1 for t in self.intervention_timestamps 
                         if current_time - t <= self.recent_intervention_window)
        return recent_count
    
    def get_intervention_observation(self):
        """Get binary observation of recent intervention"""
        current_time = datetime.now().timestamp()
        return int(any(current_time - t <= self.recent_intervention_window 
                      for t in self.intervention_timestamps))
    
    def get_current_trust_state(self):
        """Get the current trust state"""
        current_time = datetime.now().timestamp()
        return {
            'trust_level': self.trust_level,
            'recent_interventions': self.get_recent_interventions(),
            'time_since_last_intervention': current_time - self.last_intervention_time 
                if self.intervention_timestamps else float('inf')
        }
        
    def update_display(self):
        """Update the trust feedback display"""
        self.screen.fill(self.WHITE)
        
        # Draw trust level bar
        bar_width = 600
        bar_height = 30
        x = 100
        y = 50
        
        # Background bar
        pygame.draw.rect(self.screen, self.BLACK, 
                        (x, y, bar_width, bar_height), 2)
        
        # Trust level fill
        fill_width = int(bar_width * self.trust_level)
        pygame.draw.rect(self.screen, self.GREEN,
                        (x, y, fill_width, bar_height))
        
        # Draw text
        font = pygame.font.Font(None, 36)
        trust_text = f"Trust Level: {self.trust_level:.2f}"
        text_surface = font.render(trust_text, True, self.BLACK)
        self.screen.blit(text_surface, (x, y + 40))
        
        interventions_text = f"Recent Interventions: {self.get_recent_interventions()}"
        text_surface = font.render(interventions_text, True, self.BLACK)
        self.screen.blit(text_surface, (x, y + 80))
        
        pygame.display.flip()
    
    def save_session_data(self):
        """Save the trust feedback data for the session"""
        data = {
            'session_id': self.session_id,
            'manual_interventions': self.manual_interventions,
            'final_trust_level': self.trust_level,
            'intervention_count': len(self.manual_interventions)
        }
        
        filename = os.path.join(self.data_dir, f"trust_data_{self.session_id}.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    def cleanup(self):
        """Clean up pygame resources"""
        self.save_session_data()
        pygame.quit() 