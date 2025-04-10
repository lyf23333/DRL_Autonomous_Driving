import pygame
import math
import carla

def render_trust_visualization(screen, font, trust_interface, vehicle, camera_width, camera_height, trust_viz_height, reward_components, trust_history, max_trust_history, target_speed):
    """Render trust level visualization below the camera view"""
    # Draw background for trust visualization
    trust_viz_rect = pygame.Rect(0, camera_height, camera_width, trust_viz_height)
    pygame.draw.rect(screen, (20, 20, 20), trust_viz_rect)
    pygame.draw.line(screen, (50, 50, 50), (0, camera_height), (camera_width, camera_height), 1)
    
    # Draw trust level and speed on the same line
    if trust_interface:
        trust_text = font.render(f"Trust: {trust_interface.trust_level:.2f}", True, (255, 255, 255))
        screen.blit(trust_text, (10, camera_height + 10))
        
        # Indicate if intervention is active and show intervention probability
        if hasattr(trust_interface, 'intervention_active') and trust_interface.intervention_active:
            intervention_text = font.render("INTERVENTION", True, (255, 0, 0))
            screen.blit(intervention_text, (250, camera_height + 10))
            
        # Add intervention probability
        if hasattr(trust_interface, 'current_intervention_prob'):
            prob_value = trust_interface.current_intervention_prob
            # Color changes from green (low) to red (high probability)
            prob_color = (int(min(255, prob_value * 510)), int(max(0, 255 - prob_value * 510)), 0)
            prob_text = font.render(f"Interv. Prob: {prob_value:.2f}", True, prob_color)
            screen.blit(prob_text, (550, camera_height + 10))
    
    # Display current speed
    if vehicle:
        velocity = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s to km/h
        
        # Create and show speed text
        speed_text = font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255))
        screen.blit(speed_text, (150, camera_height + 10))
        
        # Add target speed in a separate section with a different color
        target_text = font.render(f"Target: {target_speed:.1f} km/h", True, (255, 200, 0))  # Yellow color
        screen.blit(target_text, (400, camera_height + 10))
        
        # Add speed compliance metric on second line
        if trust_interface and hasattr(trust_interface, 'driving_metrics') and 'speed_compliance' in trust_interface.driving_metrics:
            speed_compliance = trust_interface.driving_metrics['speed_compliance']
            
            # Color based on compliance level (red to green)
            compliance_color = (
                int(255 * (1 - speed_compliance)), 
                int(255 * speed_compliance), 
                0
            )
            
            compliance_text = font.render(f"Speed Compliance: {speed_compliance:.2f}", True, compliance_color)
            screen.blit(compliance_text, (150, camera_height + 35))
            
            # Draw compliance visual indicator (colored dot)
            pygame.draw.circle(screen, compliance_color, (130, camera_height + 40), 8)
    
    # Draw a separator line
    pygame.draw.line(
        screen, 
        (50, 50, 50), 
        (0, camera_height + 55), 
        (camera_width, camera_height + 55), 
        1
    )
    
    # Define colors and labels for each component
    component_colors = {
        'path': (0, 200, 0),       # Green
        'progress': (0, 150, 255),  # Blue
        'safety': (255, 200, 0),    # Yellow
        'comfort': (150, 0, 255),   # Purple
        'trust': (255, 150, 0),     # Orange
        'intervention': (255, 0, 0)  # Red
    }
    
    component_labels = {
        'path': 'Path Reward',
        'progress': 'Speed Reward',
        'safety': 'Safety Reward',
        'comfort': 'Comfort Reward',
        'trust': 'Trust Reward',
        'intervention': 'Intervention Penalty'
    }
    
    # Calculate dimensions for reward components display
    components_left = 10
    components_top = camera_height + 65  # Start below the separator
    component_height = 16  # Increased from 12
    component_spacing = 10  # Increased from 5
    label_width = 150
    value_width = 60  # Increased from 50
    bar_width = 150
    max_bar_value = 0.5  # Maximum expected reward value for scaling
    
    # Draw title for reward components
    title_text = font.render("REWARD COMPONENTS", True, (200, 200, 200))
    screen.blit(title_text, (components_left, components_top))
    components_top += 25  # Add space after title
    
    # Draw reward components as rows with bars
    y_pos = components_top
    for i, (component, value) in enumerate(reward_components.items()):
        # Draw the component label
        label = font.render(component_labels[component], True, (200, 200, 200))
        screen.blit(label, (components_left, y_pos))
        
        # Draw the component value
        value_text = font.render(f"{value:.2f}", True, component_colors[component])
        screen.blit(value_text, (components_left + label_width, y_pos))
        
        # Draw a bar representing the value
        bar_left = components_left + label_width + value_width
        bar_height = component_height
        
        # Draw background bar
        bg_rect = pygame.Rect(bar_left, y_pos + 2, bar_width, bar_height - 4)
        pygame.draw.rect(screen, (40, 40, 40), bg_rect)
        
        # Draw value bar
        if value != 0:
            # Scale the bar width based on the value
            scaled_width = min(abs(value) / max_bar_value * bar_width, bar_width)
            
            # For negative values, draw from the middle
            if value < 0:
                bar_rect = pygame.Rect(
                    bar_left + bar_width/2 - scaled_width, 
                    y_pos + 2, 
                    scaled_width, 
                    bar_height - 4
                )
            else:
                bar_rect = pygame.Rect(
                    bar_left + bar_width/2, 
                    y_pos + 2, 
                    scaled_width, 
                    bar_height - 4
                )
            
            pygame.draw.rect(screen, component_colors[component], bar_rect)
        
        # Draw center line for reference
        center_x = bar_left + bar_width/2
        pygame.draw.line(
            screen,
            (100, 100, 100),
            (center_x, y_pos),
            (center_x, y_pos + bar_height),
            1
        )
        
        # Move to next component
        y_pos += component_height + component_spacing
        
    # Draw trust history graph if we have data
    if trust_history:
        # Graph dimensions - adjusted for row-based layout
        graph_left = 400
        graph_width = camera_width - graph_left - 10
        graph_top = camera_height + 65  # Start below the separator
        graph_height = 140  # Fixed shorter height for the graph
        
        # Draw title for trust graph
        title_text = font.render("TRUST HISTORY", True, (200, 200, 200))
        screen.blit(title_text, (graph_left, graph_top))
        graph_top += 25  # Add space after title
        graph_height -= 25  # Adjust height to account for title
        
        # Draw graph background
        graph_rect = pygame.Rect(graph_left, graph_top, graph_width, graph_height)
        pygame.draw.rect(screen, (30, 30, 30), graph_rect)  # Darker background for graph
        pygame.draw.rect(screen, (100, 100, 100), graph_rect, 1)  # Border
        
        # Draw horizontal lines for trust levels with labels
        for i in [0, 5, 10]:
            trust_value = i * 0.1
            y = graph_top + graph_height - (trust_value * graph_height)
            pygame.draw.line(
                screen, 
                (70, 70, 70), 
                (graph_left, y), 
                (graph_left + graph_width, y), 
                1
            )
            
            # Add label for trust level
            level_text = font.render(f"{trust_value:.1f}", True, (150, 150, 150))
            screen.blit(level_text, (graph_left - 25, y - 8))
        
        # Draw trust history
        if len(trust_history) > 1:
            points = []
            for i, trust in enumerate(trust_history):
                x = graph_left + (i / (max_trust_history - 1)) * graph_width
                y = graph_top + graph_height - (trust * graph_height)
                points.append((x, y))
            
            # Draw line connecting trust points
            if len(points) > 1:
                pygame.draw.lines(screen, (0, 255, 0), False, points, 2)
            
            # Draw current trust level indicator
            current_x = graph_left + ((len(trust_history) - 1) / (max_trust_history - 1)) * graph_width
            current_y = graph_top + graph_height - (trust_history[-1] * graph_height)
            pygame.draw.circle(screen, (255, 0, 0), (int(current_x), int(current_y)), 4)
            
            # Display current trust value
            current_trust = trust_history[-1]
            current_text = font.render(f"Current: {current_trust:.2f}", True, (255, 255, 255))
            screen.blit(current_text, (graph_left + graph_width - 120, graph_top - 20))
            
            # Draw speed compliance level as a horizontal line on the graph
            if trust_interface and hasattr(trust_interface, 'driving_metrics') and 'speed_compliance' in trust_interface.driving_metrics:
                compliance_value = trust_interface.driving_metrics['speed_compliance']
                compliance_y = graph_top + graph_height - (compliance_value * graph_height)
                
                # Draw dashed horizontal line with compliance color
                compliance_color = (0, 200, 200)  # Teal color
                dash_length = 5
                gap_length = 5
                start_x = graph_left
                
                while start_x < graph_left + graph_width:
                    end_x = min(start_x + dash_length, graph_left + graph_width)
                    pygame.draw.line(
                        screen,
                        compliance_color,
                        (start_x, compliance_y),
                        (end_x, compliance_y),
                        2
                    )
                    start_x = end_x + gap_length
    
    # Add additional information at the bottom
    bottom_section_y = camera_height + trust_viz_height - 60  # Position 60px from bottom
    
    # Draw a separator line
    pygame.draw.line(
        screen, 
        (50, 50, 50), 
        (0, bottom_section_y), 
        (camera_width, bottom_section_y), 
        1
    )
    
    bottom_section_y += 10  # Add space after separator
    
    # Draw behavior adjustment info (no need to repeat target speed as it's now in the top line)
    if trust_interface and hasattr(trust_interface, 'behavior_adjustment'):
        behavior = trust_interface.behavior_adjustment
        
        # Draw a title for behavior parameters
        behavior_title = font.render("Behavior Parameters:", True, (200, 200, 200))
        screen.blit(behavior_title, (20, bottom_section_y))
        
        # Draw behavior parameters in a compact format
        left_pos = 200  # Start position for behavior values
        
        for param, value in behavior.items():
            if param == 'trust_level':
                continue  # Skip trust level as it's already shown
            
            param_name = param.replace('_', ' ').title()
            param_text = font.render(f"{param_name}: {value:.2f}", True, (200, 200, 200))
            screen.blit(param_text, (left_pos, bottom_section_y))
            left_pos += 200  # Space between parameters


def render_waypoints_on_camera(screen, sensors, camera_width, camera_height, waypoints, current_waypoint_idx, waypoint_lookahead):
    """Project and render waypoints onto the camera view"""
    
    # Get camera parameters
    camera_transform = sensors['camera'].get_transform()
    camera_location = camera_transform.location
    camera_rotation = camera_transform.rotation
    
    # Camera intrinsic parameters (approximated for a typical camera)
    fov = 90.0  # Field of view in degrees
    focal_length = camera_width / (2.0 * math.tan(math.radians(fov) / 2.0))
    
    # Calculate how many waypoints to show
    start_idx = current_waypoint_idx
    end_idx = min(start_idx + waypoint_lookahead, len(waypoints))
    
    # Draw each waypoint in the range
    for i in range(start_idx, end_idx):
        if i >= len(waypoints):
            break
            
        waypoint = waypoints[i]
        
        # Create a 3D point for the waypoint
        # Check if it's a CARLA Waypoint object or a simple waypoint with x,y attributes
        if hasattr(waypoint, 'transform'):
            # It's a CARLA Waypoint object
            waypoint_location = carla.Location(
                x=waypoint.transform.location.x,
                y=waypoint.transform.location.y,
                z=waypoint.transform.location.z + 0.5  # Slightly above ground
            )
        else:
            # It's a simple waypoint with x,y attributes
            waypoint_location = carla.Location(x=waypoint.x, y=waypoint.y, z=0.5)
        
        # Transform waypoint to camera space
        waypoint_location_world = waypoint_location
        
        # Calculate vector from camera to waypoint in world space
        to_waypoint = waypoint_location_world - camera_location
        
        # Convert to camera local coordinates
        forward = camera_rotation.get_forward_vector()
        right = camera_rotation.get_right_vector()
        up = camera_rotation.get_up_vector()
        
        # Project the vector onto camera axes
        forward_proj = to_waypoint.x * forward.x + to_waypoint.y * forward.y + to_waypoint.z * forward.z
        right_proj = to_waypoint.x * right.x + to_waypoint.y * right.y + to_waypoint.z * right.z
        up_proj = to_waypoint.x * up.x + to_waypoint.y * up.y + to_waypoint.z * up.z
        
        # Skip points behind the camera
        if forward_proj <= 0:
            continue
            
        # Project to 2D screen space
        screen_x = camera_width / 2 + focal_length * right_proj / forward_proj
        screen_y = camera_height / 2 - focal_length * up_proj / forward_proj
        
        # Skip points outside the screen
        if (screen_x < 0 or screen_x >= camera_width or 
            screen_y < 0 or screen_y >= camera_height):
            continue
            
        # Calculate marker size based on distance (closer = bigger)
        marker_size = max(5, int(20.0 / (1.0 + 0.1 * forward_proj)))
        
        # Color based on index (current waypoint is red, future ones fade to yellow)
        progress = float(i - start_idx) / max(1, waypoint_lookahead - 1)
        color = (255, int(255 * progress), 0)  # Red to yellow gradient
        
        # Draw the waypoint marker
        pygame.draw.circle(screen, color, (int(screen_x), int(screen_y)), marker_size)
        
        # Draw number for the first few waypoints
        if i < start_idx + 5:  # Only number the first 5 visible waypoints
            font = pygame.font.Font(None, 20)
            text = font.render(str(i - start_idx + 1), True, (255, 255, 255))
            screen.blit(text, (int(screen_x) - 5, int(screen_y) - 8))
