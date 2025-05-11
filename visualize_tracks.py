#!/usr/bin/env python3
"""
Visualize vehicle tracks from CARLA environment recordings.
This script loads the top-down image and track data from a recording session
and overlays the track, waypoints, and other vehicles on the image.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import cv2
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import glob
from datetime import datetime
import matplotlib.animation as animation
from matplotlib import cm

def load_recording_data(recording_dir):
    """
    Load all data from a recording directory.
    
    Args:
        recording_dir: Path to the recording directory
        
    Returns:
        dict: Recording data including image, tracks, waypoints, and metadata
    """
    data = {
        'image': None,
        'tracks': None,
        'waypoints': None,
        'other_vehicles': None,
        'metadata': None
    }
    
    # Load top-down image
    image_path = os.path.join(recording_dir, 'top_down_view.png')
    if os.path.exists(image_path):
        data['image'] = cv2.imread(image_path)
        # Convert from BGR to RGB
        data['image'] = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: No top-down image found at {image_path}")
    
    # Load track data
    track_path = os.path.join(recording_dir, 'track_data.json')
    if os.path.exists(track_path):
        with open(track_path, 'r') as f:
            data['tracks'] = json.load(f)
    else:
        print(f"Warning: No track data found at {track_path}")
    
    # Load waypoints data
    waypoints_path = os.path.join(recording_dir, 'waypoints_data.json')
    if os.path.exists(waypoints_path):
        with open(waypoints_path, 'r') as f:
            data['waypoints'] = json.load(f)
    else:
        print(f"Warning: No waypoint data found at {waypoints_path}")
    
    # Load other vehicles data
    other_vehicles_path = os.path.join(recording_dir, 'other_vehicles_data.json')
    if os.path.exists(other_vehicles_path):
        with open(other_vehicles_path, 'r') as f:
            data['other_vehicles'] = json.load(f)
    else:
        print(f"Warning: No other vehicles data found at {other_vehicles_path}")
    
    # Load metadata
    metadata_path = os.path.join(recording_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            data['metadata'] = json.load(f)
    else:
        print(f"Warning: No metadata found at {metadata_path}")
    
    return data

def plot_static_tracks(data, output_path=None, show_waypoints=True, show_other_vehicles=True):
    """
    Plot the tracks on top of the top-down image.
    
    Args:
        data: Recording data including image, tracks, and waypoints
        output_path: Path to save the visualization
        show_waypoints: Whether to show waypoints on the visualization
        show_other_vehicles: Whether to show other vehicles on the visualization
    """
    if data['image'] is None or data['tracks'] is None:
        print("Error: Image or track data is missing")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display the image
    ax.imshow(data['image'])
    
    # Extract track coordinates
    track_x = [point['x'] for point in data['tracks']]
    track_y = [point['y'] for point in data['tracks']]
    speeds = [point['speed'] for point in data['tracks']]
    
    # Create a colormap for speed
    norm = plt.Normalize(min(speeds), max(speeds))
    cmap = plt.get_cmap('plasma')
    
    # Create line segments for speed color gradient
    points = np.array([track_x, track_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(speeds))
    lc.set_linewidth(3)
    ax.add_collection(lc)
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Speed (m/s)')
    
    # Plot start and end points
    ax.plot(track_x[0], track_y[0], 'go', markersize=10, label='Start')
    ax.plot(track_x[-1], track_y[-1], 'ro', markersize=10, label='End')
    
    # Plot waypoints if available and requested
    if show_waypoints and data['waypoints'] is not None:
        waypoint_x = [point['x'] for point in data['waypoints']]
        waypoint_y = [point['y'] for point in data['waypoints']]
        ax.plot(waypoint_x, waypoint_y, 'w--', linewidth=2, alpha=0.7, label='Waypoints')
        
        # Add waypoint indices
        for i, (x, y) in enumerate(zip(waypoint_x, waypoint_y)):
            if i % 5 == 0:  # Only label every 5th waypoint to avoid crowding
                txt = ax.text(x, y, str(i), color='white', fontsize=10, 
                             horizontalalignment='center', verticalalignment='center')
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Plot other vehicles if available and requested
    if show_other_vehicles and data['other_vehicles'] is not None:
        # Create a dictionary to track the last known position of each vehicle
        vehicle_positions = {}
        
        for frame in data['other_vehicles']:
            for vehicle in frame:
                vehicle_id = vehicle['id']
                vehicle_positions[vehicle_id] = (vehicle['x'], vehicle['y'])
        
        # Plot the last known position of each vehicle
        for vehicle_id, (x, y) in vehicle_positions.items():
            ax.plot(x, y, 'bx', markersize=8)
    
    # Set title and labels
    title = "Vehicle Track Visualization"
    if data['metadata'] is not None:
        title += f" ({data['metadata']['town']})"
    ax.set_title(title)
    
    # Remove axes ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Add metadata info
    if data['metadata'] is not None:
        info_text = (
            f"Town: {data['metadata']['town']}\n"
            f"Track points: {data['metadata']['num_track_points']}\n"
            f"Waypoints: {data['metadata']['num_waypoints']}"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=10)
    
    # Save the visualization if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def create_animated_visualization(data, output_path=None, fps=30, show_waypoints=True, show_other_vehicles=True):
    """
    Create an animated visualization of the vehicle's path.
    
    Args:
        data: Recording data including image, tracks, and waypoints
        output_path: Path to save the animation
        fps: Frames per second for the animation
        show_waypoints: Whether to show waypoints in the animation
        show_other_vehicles: Whether to show other vehicles in the animation
    """
    if data['image'] is None or data['tracks'] is None:
        print("Error: Image or track data is missing")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display the image
    ax.imshow(data['image'])
    
    # Extract track coordinates
    track_x = np.array([point['x'] for point in data['tracks']])
    track_y = np.array([point['y'] for point in data['tracks']])
    speeds = np.array([point['speed'] for point in data['tracks']])
    
    # Plot waypoints if available and requested
    if show_waypoints and data['waypoints'] is not None:
        waypoint_x = [point['x'] for point in data['waypoints']]
        waypoint_y = [point['y'] for point in data['waypoints']]
        ax.plot(waypoint_x, waypoint_y, 'w--', linewidth=2, alpha=0.5, label='Waypoints')
    
    # Create a colormap for speed
    norm = plt.Normalize(min(speeds), max(speeds))
    cmap = plt.get_cmap('plasma')
    
    # Initialize plot elements
    trail_line, = ax.plot([], [], 'g-', linewidth=3, alpha=0.7, label='Trail')
    head_point, = ax.plot([], [], 'ro', markersize=8, label='Vehicle')
    speed_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                        backgroundcolor='black', color='white')
    
    # Set title and labels
    title = "Vehicle Track Animation"
    if data['metadata'] is not None:
        title += f" ({data['metadata']['town']})"
    ax.set_title(title)
    
    # Remove axes ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Speed (m/s)')
    
    # Initialize other vehicle markers if requested
    vehicle_markers = {}
    if show_other_vehicles and data['other_vehicles'] is not None:
        # Create a flat list of all vehicle IDs across all frames
        all_vehicle_ids = set()
        for frame in data['other_vehicles']:
            for vehicle in frame:
                all_vehicle_ids.add(vehicle['id'])
        
        # Create a marker for each vehicle
        for vehicle_id in all_vehicle_ids:
            marker, = ax.plot([], [], 'bx', markersize=8)
            vehicle_markers[vehicle_id] = marker
    
    def init():
        trail_line.set_data([], [])
        head_point.set_data([], [])
        speed_text.set_text('')
        
        # Initialize all vehicle markers
        for marker in vehicle_markers.values():
            marker.set_data([], [])
        
        return [trail_line, head_point, speed_text] + list(vehicle_markers.values())
    
    def animate(i):
        # Update trail (fixed length of 50 points)
        start_idx = max(0, i - 50)
        trail_line.set_data(track_x[start_idx:i+1], track_y[start_idx:i+1])
        
        # Update vehicle position
        head_point.set_data(track_x[i], track_y[i])
        
        # Update color based on speed
        color = cmap(norm(speeds[i]))
        head_point.set_color(color)
        trail_line.set_color(color)
        
        # Update speed text
        speed_text.set_text(f'Speed: {speeds[i]:.2f} m/s')
        
        # Update other vehicles if available
        if show_other_vehicles and data['other_vehicles'] is not None:
            # Find the closest frame of other vehicles data
            frame_idx = min(i, len(data['other_vehicles']) - 1)
            
            # Reset all markers
            for marker in vehicle_markers.values():
                marker.set_data([], [])
            
            # Update marker positions for this frame
            if frame_idx < len(data['other_vehicles']):
                for vehicle in data['other_vehicles'][frame_idx]:
                    vehicle_id = vehicle['id']
                    if vehicle_id in vehicle_markers:
                        marker = vehicle_markers[vehicle_id]
                        marker.set_data(vehicle['x'], vehicle['y'])
        
        return [trail_line, head_point, speed_text] + list(vehicle_markers.values())
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(track_x), interval=1000/fps,
                                  blit=True)
    
    # Save animation if requested
    if output_path:
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
        print(f"Animation saved to {output_path}")
    
    # Show the animation
    plt.tight_layout()
    plt.show()

def list_recording_sessions(base_dir="recordings"):
    """
    List all available recording sessions.
    
    Args:
        base_dir: Base directory containing recording sessions
        
    Returns:
        list: List of recording session directories
    """
    sessions = []
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Recording directory {base_dir} does not exist")
        return sessions
    
    # Find all session directories
    session_dirs = glob.glob(os.path.join(base_dir, "session_*"))
    
    for session_dir in sorted(session_dirs):
        # Check if this is a valid session with at least track data
        track_path = os.path.join(session_dir, 'track_data.json')
        if os.path.exists(track_path):
            # Get session timestamp from directory name
            session_name = os.path.basename(session_dir)
            try:
                timestamp_str = session_name.replace("session_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_timestamp = session_name
            
            # Get basic info about the session
            metadata_path = os.path.join(session_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                town = metadata.get('town', 'Unknown')
                num_track_points = metadata.get('num_track_points', 0)
            else:
                town = "Unknown"
                # Count track points directly from the track data
                with open(track_path, 'r') as f:
                    track_data = json.load(f)
                num_track_points = len(track_data)
            
            sessions.append({
                'dir': session_dir,
                'name': session_name,
                'timestamp': formatted_timestamp,
                'town': town,
                'num_track_points': num_track_points,
                'has_image': os.path.exists(os.path.join(session_dir, 'top_down_view.png'))
            })
    
    return sessions

def main():
    parser = argparse.ArgumentParser(description='Visualize vehicle tracks from CARLA environment recordings')
    parser.add_argument('--recording-dir', type=str, default=None,
                        help='Path to the recording directory (default: most recent session)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the visualization')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available recording sessions and exit')
    parser.add_argument('--animate', '-a', action='store_true',
                        help='Create an animated visualization')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for animation (default: 30)')
    parser.add_argument('--no-waypoints', action='store_true',
                        help='Hide waypoints from visualization')
    parser.add_argument('--no-other-vehicles', action='store_true',
                        help='Hide other vehicles from visualization')
    args = parser.parse_args()
    
    # List available sessions if requested
    if args.list:
        sessions = list_recording_sessions()
        if sessions:
            print("Available recording sessions:")
            for i, session in enumerate(sessions):
                image_status = "with image" if session['has_image'] else "no image"
                print(f"{i+1}. {session['timestamp']} - {session['town']} - {session['num_track_points']} points ({image_status})")
            print(f"\nUse --recording-dir DIRECTORY to visualize a specific session")
        else:
            print("No recording sessions found")
        return
    
    # Determine which recording to visualize
    if args.recording_dir is None:
        # Use the most recent session
        sessions = list_recording_sessions()
        if not sessions:
            print("Error: No recording sessions found")
            return
        recording_dir = sessions[-1]['dir']
        print(f"Using most recent recording: {sessions[-1]['timestamp']}")
    else:
        recording_dir = args.recording_dir
    
    # Load recording data
    data = load_recording_data(recording_dir)
    
    # Create visualization
    show_waypoints = not args.no_waypoints
    show_other_vehicles = not args.no_other_vehicles
    
    if args.animate:
        output_path = args.output if args.output else os.path.join(recording_dir, 'track_animation.mp4')
        create_animated_visualization(data, output_path, args.fps, show_waypoints, show_other_vehicles)
    else:
        output_path = args.output if args.output else os.path.join(recording_dir, 'track_visualization.png')
        plot_static_tracks(data, output_path, show_waypoints, show_other_vehicles)

if __name__ == "__main__":
    main() 