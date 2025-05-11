#!/usr/bin/env python3
"""
Plot data from TensorBoard CSV files on the same graph with a focus on
plotting the 'Value' column against the 'Step' column.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path
import sys

# Define algorithm colors as a global constant for consistency across functions
ALGORITHM_COLORS = {
    'DDPG': '#1f77b4',  # Blue
    'DQN': '#ff7f0e',   # Orange
    'SAC': '#2ca02c',   # Green
    'PPO': '#d62728',   # Red
    'TD3': '#9467bd',   # Purple
    'A2C': '#8c564b',   # Brown
    'TRPO': '#e377c2',  # Pink
    'ACER': '#7f7f7f',  # Gray
    'ACKTR': '#bcbd22', # Yellow-green
    'GAIL': '#17becf'   # Cyan
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot TensorBoard CSV data.")
    parser.add_argument('--files', '-f', nargs='+', 
                      help="Paths to specific CSV files (default: all files in tensorboard_csv_files directory)")
    parser.add_argument('--title', '-t', type=str, default="TensorBoard Data Comparison",
                      help="Title for the plot")
    parser.add_argument('--x-label', '-xl', type=str, default="Step",
                      help="Label for the x-axis")
    parser.add_argument('--y-label', '-yl', type=str, default="Value",
                      help="Label for the y-axis")
    parser.add_argument('--legends', '-l', nargs='+', 
                      help="Custom legend names for each file")
    parser.add_argument('--output', '-o', type=str, default="tensorboard_plot.png",
                      help="Output file path for the saved plot")
    parser.add_argument('--styles', '-s', nargs='+', 
                      choices=['solid', 'dashed', 'dotted', 'dashdot'],
                      help="Line styles for each dataset")
    parser.add_argument('--colors', '-c', nargs='+', 
                      help="Colors for each dataset")
    parser.add_argument('--markers', '-m', nargs='+', 
                      choices=['o', 's', '^', 'v', '<', '>', 'p', '*', '+', 'x', 'D', 'd', 'None'],
                      help="Markers for each dataset (use 'None' for no markers)")
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 8],
                      help="Figure size in inches (width height)")
    parser.add_argument('--smooth', '-sm', type=int, default=15,
                      help="Apply smoothing with specified window size (0 for no smoothing)")
    parser.add_argument('--smooth-method', type=str, choices=['rolling', 'exponential', 'savgol'], default='rolling',
                      help="Smoothing method to use: rolling (moving average), exponential (EMA), or savgol (Savitzky-Golay filter)")
    parser.add_argument('--normalize', '-n', action='store_true',
                      help="Normalize Steps to start from 0 for easier comparison")
    parser.add_argument('--grid', '-g', action='store_true', default=False,
                      help="Show grid on the plot (default: False)")
    parser.add_argument('--y-min', type=float, default=None,
                      help="Minimum value for y-axis")
    parser.add_argument('--y-max', type=float, default=None,
                      help="Maximum value for y-axis")
    parser.add_argument('--generate-color-legend', action='store_true', 
                      help="Generate a reference image showing algorithm color mappings")
    parser.add_argument('--color-legend-path', type=str, default="algorithm_colors.png",
                      help="Path to save the color legend image")
    return parser.parse_args()


def read_tensorboard_csv(file_path, smooth_window=15, smooth_method='rolling', normalize=False):
    """
    Read data from a TensorBoard CSV file.
    
    Args:
        file_path: Path to the CSV file
        smooth_window: Window size for smoothing (0 for no smoothing)
        smooth_method: Method for smoothing ('rolling', 'exponential', or 'savgol')
        normalize: If True, normalize steps to start from 0
        
    Returns:
        Tuple of (steps, values)
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure the required columns exist
        if 'Step' not in df.columns or 'Value' not in df.columns:
            print(f"Error: Required columns 'Step' and 'Value' not found in {file_path}")
            return None, None
        
        # Extract the step and value columns
        steps = df['Step'].values
        values = df['Value'].values
        
        # Normalize steps if requested
        if normalize and len(steps) > 0:
            steps = steps - steps[0]
        
        # Apply smoothing if requested
        if smooth_window > 0 and len(values) > smooth_window:
            # Create a pandas Series for easy smoothing
            values_series = pd.Series(values)
            
            if smooth_method == 'rolling':
                # Apply centered rolling average
                smoothed = values_series.rolling(window=smooth_window, center=True).mean()
                
                # Fill NaN values that occur at the beginning and end of the smoothed series
                # with the original values
                values = smoothed.fillna(values_series).values
                
            elif smooth_method == 'exponential':
                # Apply exponential moving average
                # alpha=2/(window+1) is a common way to relate EMA to window size
                alpha = 2 / (smooth_window + 1)
                smoothed = values_series.ewm(alpha=alpha, adjust=False).mean()
                values = smoothed.values
                
            elif smooth_method == 'savgol':
                try:
                    from scipy.signal import savgol_filter
                    # For Savitzky-Golay filter, window must be odd
                    window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
                    # Apply Savitzky-Golay filter (polynomial order 3 is common)
                    values = savgol_filter(values, window, 3)
                except ImportError:
                    print("Warning: scipy not available for Savitzky-Golay filter, falling back to rolling average")
                    smoothed = values_series.rolling(window=smooth_window, center=True).mean()
                    values = smoothed.fillna(values_series).values
        
        return steps, values
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None


def extract_algorithm_name(file_path):
    """
    Extract the algorithm name from the TensorBoard file path.
    
    Args:
        file_path: Path to the TensorBoard CSV file
        
    Returns:
        String containing the algorithm name if found, otherwise the filename
    """
    filename = os.path.basename(file_path)
    
    # TensorBoard filenames typically follow a pattern like:
    # run-YYYY-MM-DD_HH-MM-SS_[experiment_name]-tag-[metric_name].csv
    
    # Try to extract experiment name
    try:
        # Split on 'run-' and then take everything until '-tag-'
        run_info = filename.split('run-')[1].split('-tag-')[0]
        
        # Find algorithm name - usually after 'render_' or at the end
        algorithms = ['ddpg', 'dqn', 'sac', 'ppo', 'td3']
        
        for algo in algorithms:
            if algo in run_info:
                # Get the context around the algorithm
                parts = run_info.split(algo)
                if len(parts) > 1:
                    # Check if algo is preceded by _ or whitespace
                    if parts[0].endswith('_') or parts[0].endswith(' '):
                        return algo.upper()
        
        # If no specific algorithm found, use the experiment name part
        parts = run_info.split('_')
        if len(parts) > 3:  # Skip date/time parts
            return '_'.join(parts[2:])
        
        return run_info
    except:
        # Fallback to filename without extension
        return Path(file_path).stem


def generate_color_legend(output_path="algorithm_colors.png"):
    """
    Generate a reference image showing the mapping between algorithms and their standard colors.
    
    Args:
        output_path: Path to save the legend image
    """
    plt.figure(figsize=(10, 6))
    
    # Define the desired order
    desired_order = ['DDPG', 'DQN', 'PPO', 'SAC']
    
    # Create ordered lists of algorithms and colors
    ordered_algorithms = []
    ordered_colors = []
    
    # First add algorithms in the desired order
    for algo in desired_order:
        if algo in ALGORITHM_COLORS:
            ordered_algorithms.append(algo)
            ordered_colors.append(ALGORITHM_COLORS[algo])
    
    # Then add any remaining algorithms
    for algo, color in ALGORITHM_COLORS.items():
        if algo not in desired_order:
            ordered_algorithms.append(algo)
            ordered_colors.append(color)
    
    # Create a horizontal bar chart with algorithms in desired order
    y_pos = np.arange(len(ordered_algorithms))
    plt.barh(y_pos, [1] * len(ordered_algorithms), color=ordered_colors)
    
    # Add algorithm names as y-tick labels
    plt.yticks(y_pos, ordered_algorithms, fontsize=14)
    
    # Add title and labels
    plt.title("Standard Algorithm Color Mapping", fontsize=16)
    plt.xlabel("", fontsize=14)
    plt.ylabel("Algorithm", fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Color legend saved to {output_path}")
    plt.close()


def plot_tensorboard_data(file_paths, title, x_label, y_label, legends, output_path, 
                         styles, colors, markers, figsize, smooth_window, smooth_method, normalize, grid, y_min=None, y_max=None):
    """
    Plot TensorBoard data from multiple files on the same graph.
    
    Args:
        file_paths: List of file paths to TensorBoard CSV files
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        legends: List of legend labels
        output_path: Path to save the plot
        styles: List of line styles
        colors: List of colors
        markers: List of markers
        figsize: Figure size as (width, height)
        smooth_window: Window size for smoothing
        smooth_method: Method for smoothing ('rolling', 'exponential', or 'savgol')
        normalize: Whether to normalize steps to start from 0
        grid: Whether to show grid on the plot
        y_min: Minimum value for y-axis (None for auto-scaling)
        y_max: Maximum value for y-axis (None for auto-scaling)
    """
    plt.figure(figsize=figsize)
    
    # Define the desired legend order
    desired_order = ['DDPG', 'DQN', 'PPO', 'SAC']
    
    # Default styles if not provided
    default_styles = ['solid', 'solid', 'solid', 'solid']
    
    # Use global algorithm colors
    algorithm_colors = ALGORITHM_COLORS
    
    # Default fallback colors for unknown algorithms
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Default markers
    default_markers = ['None', 'o', 's', '^', 'v']
    
    # First extract all algorithm names to determine consistent coloring
    all_algorithms = []
    for i, file_path in enumerate(file_paths):
        if legends and i < len(legends):
            algo = legends[i]
        else:
            algo = extract_algorithm_name(file_path)
        all_algorithms.append(algo)
    
    # Store plot lines and labels for later ordering
    plot_lines = []
    plot_labels = []
    
    # Now plot each dataset
    for i, file_path in enumerate(file_paths):
        # Read data
        steps, values = read_tensorboard_csv(file_path, smooth_window, smooth_method, normalize)
        
        if steps is None or values is None:
            continue
            
        # Use provided styles or defaults
        style = styles[i] if styles and i < len(styles) else default_styles[i % len(default_styles)]
        
        # Get algorithm name for this file
        if legends and i < len(legends):
            algorithm = legends[i]
        else:
            algorithm = extract_algorithm_name(file_path)
        
        # Determine color based on algorithm name for consistency
        if colors and i < len(colors):
            # If user explicitly provided colors, use those
            color = colors[i]
        else:
            # Otherwise use our algorithm-based color mapping
            color = algorithm_colors.get(algorithm.upper(), default_colors[i % len(default_colors)])
        
        # Handle the case where marker might be 'None' string
        if markers and i < len(markers):
            marker = None if markers[i] == 'None' else markers[i]
        else:
            marker = None if default_markers[i % len(default_markers)] == 'None' else default_markers[i % len(default_markers)]
        
        # Determine markevery parameter based on data length
        # Use fewer markers for larger datasets for better visibility
        if marker is not None:
            markevery = max(1, len(steps) // 30)
        else:
            markevery = 1
            
        # Plot the data but don't add label yet
        line = plt.plot(steps, values, linestyle=style, color=color, marker=marker, 
                markersize=5, markevery=markevery, label='_nolegend_', linewidth=2)[0]
        
        # Store line and label for later ordering
        plot_lines.append(line)
        plot_labels.append(algorithm)
    
    # Set the title and labels
    # plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=20)  # Increased font size
    plt.ylabel(y_label, fontsize=20)  # Increased font size
    
    # Format x-axis ticks to show values in millions
    ax = plt.gca()
    
    # Format x-axis with millions
    def millions_formatter(x, pos):
        return f"{x/1e6:.1f}M"
        
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    
    # Increase tick label font sizes
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set y-axis limits if provided
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)
    
    # Remove grid (regardless of grid parameter)
    plt.grid(False)
    
    # Make plot background white
    ax.set_facecolor('white')
    
    # Create ordered legend
    # First, create a mapping from algorithm names to line objects
    line_map = dict(zip(plot_labels, plot_lines))
    
    # Create ordered handles and labels for legend
    ordered_handles = []
    ordered_labels = []
    
    # First add algorithms in the desired order
    for algo in desired_order:
        for label in plot_labels:
            if label.upper() == algo:
                ordered_handles.append(line_map[label])
                ordered_labels.append(label)
    
    # Then add any remaining algorithms not in the desired order
    for label in plot_labels:
        if label.upper() not in desired_order:
            ordered_handles.append(line_map[label])
            ordered_labels.append(label)
    
    # Add legend with ordered handles and labels
    plt.legend(ordered_handles, ordered_labels, loc='best', fontsize=14)
    
    # Add a tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()


def main():
    args = parse_args()
    
    # Add a new command-line argument to generate the color legend
    if '--generate-color-legend' in sys.argv:
        generate_color_legend()
        return
    
    # If no files specified, use all CSV files in the tensorboard_csv_files directory
    if not args.files:
        file_paths = glob.glob('tensorboard_csv_files/*.csv')
        if not file_paths:
            print("Error: No CSV files found in tensorboard_csv_files directory")
            return
    else:
        file_paths = args.files
        
    # Verify files exist
    valid_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not valid_files:
        print("Error: No valid files to plot")
        return
    
    # Set default axis labels for this specific use case
    args.x_label = "Step"
    args.y_label = "Acceleration Smoothness"
    args.output = "acceleration_smoothness.png"
    args.title = "TensorBoard Data Comparison"
    args.smooth_method = "savgol"
    args.smooth = 100
    
    # Plot the data
    plot_tensorboard_data(
        valid_files, 
        args.title, 
        args.x_label, 
        args.y_label, 
        args.legends, 
        args.output, 
        args.styles, 
        args.colors, 
        args.markers, 
        args.figsize,
        args.smooth,
        args.smooth_method,
        args.normalize,
        args.grid,
        y_min=0.0,
        y_max=1.0,
    )


if __name__ == "__main__":
    main() 