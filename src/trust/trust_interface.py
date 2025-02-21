import pygame
import numpy as np
import json
from datetime import datetime
import os

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
        self.manual_interventions = []
        self.intervention_timestamps = []
        
        # Setup data logging
        self.data_dir = "data/trust_feedback"
        os.makedirs(self.data_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        
        interventions_text = f"Manual Interventions: {len(self.manual_interventions)}"
        text_surface = font.render(interventions_text, True, self.BLACK)
        self.screen.blit(text_surface, (x, y + 80))
        
        pygame.display.flip()
        
    def handle_events(self):
        """Handle pygame events for trust feedback"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # Manual intervention recorded with spacebar
                if event.key == pygame.K_SPACE:
                    self.record_intervention()
                # Trust level adjustment with arrow keys
                elif event.key == pygame.K_UP:
                    self.trust_level = min(1.0, self.trust_level + 0.1)
                elif event.key == pygame.K_DOWN:
                    self.trust_level = max(0.0, self.trust_level - 0.1)
                    
        return True
    
    def record_intervention(self):
        """Record a manual intervention"""
        timestamp = datetime.now().timestamp()
        self.manual_interventions.append({
            'timestamp': timestamp,
            'trust_level': self.trust_level
        })
        self.intervention_timestamps.append(timestamp)
        
        # Decrease trust level on intervention
        self.trust_level = max(0.0, self.trust_level - 0.1)
        
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
            
    def get_current_trust_state(self):
        """Get the current trust state for the RL agent"""
        return {
            'trust_level': self.trust_level,
            'recent_interventions': len(self.intervention_timestamps),
            'time_since_last_intervention': self._time_since_last_intervention()
        }
        
    def _time_since_last_intervention(self):
        """Calculate time since last intervention"""
        if not self.intervention_timestamps:
            return float('inf')
        
        current_time = datetime.now().timestamp()
        return current_time - self.intervention_timestamps[-1]
        
    def cleanup(self):
        """Clean up pygame resources"""
        self.save_session_data()
        pygame.quit() 