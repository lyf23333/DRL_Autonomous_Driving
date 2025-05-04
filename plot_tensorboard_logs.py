#!/usr/bin/env python3
"""
Plot TensorBoard logs using Seaborn for better visualization.
This script reads TensorBoard event files and creates publication-quality plots.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import re
import multiprocessing as mp
from functools import partial
import pickle
import hashlib
import time

# Set style parameters
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS = ['o', 's', '^', 'D', 'X']

def extract_algorithm_from_path(path):
    """Extract algorithm name from tensorboard log path"""
    # Expected format: timestamp_run-name_algorithm_scenario
    parts = os.path.basename(path).split('_')
    # Find which part is the algorithm (should be after run-name)
    for part in parts:
        if part.lower() in ['ppo', 'sac', 'ddpg', 'dqn']:
            return part.upper()
    return "Unknown"

def extract_run_name(path):
    """Extract run name from tensorboard log path"""
    # Extract everything after the timestamp and before algorithm
    base_name = os.path.basename(path)
    # Remove the timestamp part (first 19 characters typically)
    if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', base_name[:19]):
        without_timestamp = base_name[20:]
        # Now extract everything before the algorithm
        for algo in ['ppo', 'sac', 'ddpg', 'dqn']:
            if f"_{algo}_" in without_timestamp.lower():
                return without_timestamp.split(f"_{algo}_")[0]
    # Fallback to directory name if pattern not found
    return os.path.basename(path)

def get_cache_path(log_path, tags):
    """Generate a cache file path based on log path and requested tags"""
    # Create a unique identifier based on the log path and tags
    if tags:
        tag_str = "_".join(sorted(tags))
    else:
        tag_str = "all_tags"
    
    # Create a hash of the tag string to keep filename reasonable
    tag_hash = hashlib.md5(tag_str.encode()).hexdigest()[:10]
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache filename based on the log directory name and tag hash
    log_dir_name = os.path.basename(os.path.normpath(log_path))
    cache_filename = f"{log_dir_name}_{tag_hash}.pkl"
    
    return os.path.join(cache_dir, cache_filename)

def is_cache_valid(cache_path, log_path):
    """Check if cache is valid by comparing modification times"""
    if not os.path.exists(cache_path):
        return False
    
    cache_mtime = os.path.getmtime(cache_path)
    
    # Check if any event file in the log directory is newer than the cache
    event_files = []
    for root, _, files in os.walk(log_path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    for event_file in event_files:
        if os.path.getmtime(event_file) > cache_mtime:
            return False
    
    return True

def process_log_directory(log_path, tags=None, max_reload_size=None, use_cache=True):
    """
    Process a single TensorBoard log directory.
    
    Args:
        log_path: Path to the log directory
        tags: Specific scalar tags to extract
        max_reload_size: Maximum size for the reload; smaller loads faster but may miss data
        use_cache: Whether to use caching to speed up loading
        
    Returns:
        Tuple of (run_name, algorithm, DataFrame) for the processed data
    """
    run_name = extract_run_name(log_path)
    algorithm = extract_algorithm_from_path(log_path)
    combined_name = f"{algorithm}: {run_name}"
    print(f"Processing {combined_name}...")
    
    # Check if we have a valid cache
    cache_path = get_cache_path(log_path, tags)
    if use_cache and is_cache_valid(cache_path, log_path):
        print(f"Using cached data for {combined_name}")
        try:
            with open(cache_path, 'rb') as f:
                run_df = pickle.load(f)
                run_df['algorithm'] = algorithm
                run_df['run_name'] = run_name
                return combined_name, run_df
        except Exception as e:
            print(f"Error loading cache: {e}, will process from scratch")
    
    # Find event file
    event_files = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_path}")
        return combined_name, None
    
    # Use the most recent event file if multiple exist
    event_file = sorted(event_files)[-1]
    
    # Set up the event accumulator with size limits if specified
    if max_reload_size:
        ea = EventAccumulator(event_file, size_guidance={
            'scalars': max_reload_size,
            'histograms': 0,  # Skip histograms
            'images': 0,      # Skip images
            'audio': 0,       # Skip audio
            'compressedHistograms': 0,
            'tensors': 0      # Skip tensors
        })
    else:
        ea = EventAccumulator(event_file)
    
    start_time = time.time()
    ea.Reload()
    reload_time = time.time() - start_time
    print(f"Reload took {reload_time:.2f} seconds")
    
    available_tags = ea.Tags()['scalars']
    
    # Only print available tags if we're looking for specific ones
    if tags:
        matching_tags = [tag for tag in available_tags if any(t in tag for t in tags)]
        if matching_tags:
            print(f"Found {len(matching_tags)} matching tags out of {len(available_tags)} available")
        else:
            print(f"None of the specified tags found in {log_path} (available: {len(available_tags)})")
            return combined_name, None
    
    # Filter tags if specified
    if tags:
        tags_to_load = [tag for tag in available_tags if any(t in tag for t in tags)]
    else:
        tags_to_load = available_tags
    
    if not tags_to_load:
        return combined_name, None
    
    # Create DataFrame for this run
    run_df = pd.DataFrame()
    
    # Process tags
    for tag in tags_to_load:
        tag_values = []
        steps = []
        
        # Extract events
        events = ea.Scalars(tag)
        for event in events:
            steps.append(event.step)
            tag_values.append(event.value)
        
        if not steps:
            continue
            
        # Create tag DataFrame
        tag_df = pd.DataFrame({
            'step': steps,
            tag: tag_values
        })
        
        # Merge with run DataFrame
        if run_df.empty:
            run_df = tag_df
        else:
            run_df = pd.merge(run_df, tag_df, on='step', how='outer')
    
    # Save to cache if requested
    if use_cache and not run_df.empty:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(run_df, f)
            print(f"Saved data to cache: {cache_path}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    if not run_df.empty:
        run_df['algorithm'] = algorithm
        run_df['run_name'] = run_name
        return combined_name, run_df
    
    return combined_name, None

def read_tensorboard_logs(log_dir, tags=None, max_reload_size=100000, use_cache=True, parallel=True):
    """
    Read TensorBoard logs from the specified directory.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        tags: Specific scalar tags to extract (if None, extract all scalar tags)
        max_reload_size: Maximum number of scalar events to load per tag (faster with lower values)
        use_cache: Whether to use caching to speed up loading
        parallel: Whether to use parallel processing
        
    Returns:
        Dictionary mapping run names to DataFrames of their metrics
    """
    print(f"Reading TensorBoard logs from {log_dir}")
    log_paths = glob.glob(os.path.join(log_dir, "*"))
    log_paths = [p for p in log_paths if os.path.isdir(p)]
    
    if not log_paths:
        print(f"No TensorBoard logs found in {log_dir}")
        return {}
    
    run_data = {}
    
    if parallel and len(log_paths) > 1:
        # Use multiprocessing for parallel processing
        num_cpus = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        print(f"Using {num_cpus} CPUs for parallel processing of {len(log_paths)} log directories")
        
        # Create a pool of workers
        with mp.Pool(processes=num_cpus) as pool:
            # Process each log directory in parallel
            process_func = partial(process_log_directory, tags=tags, 
                                max_reload_size=max_reload_size, use_cache=use_cache)
            results = pool.map(process_func, log_paths)
            
            # Collect results
            for combined_name, df in results:
                if df is not None:
                    run_data[combined_name] = df
    else:
        # Sequential processing
        for log_path in log_paths:
            combined_name, df = process_log_directory(log_path, tags, max_reload_size, use_cache)
            if df is not None:
                run_data[combined_name] = df
    
    return run_data

def smooth_data(data, weight=0.8):
    """Apply exponential moving average smoothing"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * data[i]
    return smoothed

def plot_single_metric(run_data, metric, output_dir=None, smoothing=0.8, trust_levels=None):
    """
    Plot a single metric for multiple runs.
    
    Args:
        run_data: Dictionary mapping run names to DataFrames of their metrics
        metric: Metric to plot
        output_dir: Optional directory to save the figure
        smoothing: Smoothing factor for the curve (0 = no smoothing, 1 = maximum smoothing)
        trust_levels: Optional dictionary mapping run names to trust levels for annotation
    """
    plt.figure(figsize=(12, 7))
    
    for i, (name, df) in enumerate(run_data.items()):
        if metric not in df.columns:
            print(f"Metric {metric} not found in {name}")
            continue
        
        # Extract data
        steps = df['step'].values
        values = df[metric].values
        
        # Skip runs with no data
        if len(values) == 0:
            continue
            
        # Apply smoothing
        if smoothing > 0:
            values = smooth_data(values, smoothing)
        
        # Plot the curve
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        plt.plot(steps, values, label=name, color=color, 
                 marker=marker, markevery=max(1, len(steps)//10), markersize=8)
        
        # Add trust level annotation if provided
        if trust_levels and name in trust_levels:
            plt.annotate(f"Trust: {trust_levels[name]}", 
                     xy=(steps[-1], values[-1]), 
                     xytext=(10, 0),
                     textcoords="offset points",
                     ha="left", va="center",
                     color=color,
                     fontweight="bold")
    
    # Clean up the plot
    plt.xlabel('Training Steps')
    plt.ylabel(metric.replace('/', ' - ').title())
    plt.title(f'{metric.replace("/", " - ").title()} During Training')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Format y-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt_file = os.path.join(output_dir, f"{metric.replace('/', '_')}.png")
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {plt_file}")
    
    plt.show()

def plot_multiple_metrics(run_data, metrics, output_dir=None, smoothing=0.8):
    """
    Create a subplot figure with multiple metrics.
    
    Args:
        run_data: Dictionary mapping run names to DataFrames of their metrics
        metrics: List of metrics to plot
        output_dir: Optional directory to save the figure
        smoothing: Smoothing factor for the curves
    """
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, (name, df) in enumerate(run_data.items()):
            if metric not in df.columns:
                continue
            
            # Extract data
            steps = df['step'].values
            values = df[metric].values
            
            # Skip runs with no data
            if len(values) == 0:
                continue
                
            # Apply smoothing
            if smoothing > 0:
                values = smooth_data(values, smoothing)
            
            # Plot the curve
            color = COLORS[j % len(COLORS)]
            marker = MARKERS[j % len(MARKERS)]
            ax.plot(steps, values, label=name, color=color, 
                    marker=marker, markevery=max(1, len(steps)//10), markersize=6)
        
        # Clean up the subplot
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric.replace('/', ' - ').title())
        ax.set_title(f'{metric.replace("/", " - ").title()}')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Remove any unused subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=min(5, len(run_data)))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    
    # Save if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt_file = os.path.join(output_dir, "multiple_metrics.png")
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {plt_file}")
    
    plt.show()

def plot_comparison_across_trust(run_data, metric, trust_level_mapping, output_dir=None):
    """
    Plot a comparison of a metric across different trust levels.
    
    Args:
        run_data: Dictionary mapping run names to DataFrames of their metrics
        metric: Metric to compare
        trust_level_mapping: Dictionary mapping run names to trust levels
        output_dir: Optional directory to save the figure
    """
    # Organize data by algorithm and trust level
    algorithm_data = {}
    
    for name, df in run_data.items():
        if metric not in df.columns:
            continue
            
        if name not in trust_level_mapping:
            continue
            
        algorithm = df['algorithm'].iloc[0]
        trust_level = trust_level_mapping[name]
        
        # Get the last value of the metric (final performance)
        final_value = df[metric].iloc[-1]
        
        if algorithm not in algorithm_data:
            algorithm_data[algorithm] = {'trust_levels': [], 'values': []}
            
        algorithm_data[algorithm]['trust_levels'].append(trust_level)
        algorithm_data[algorithm]['values'].append(final_value)
    
    # Plot the comparison
    plt.figure(figsize=(10, 6))
    
    for i, (algorithm, data) in enumerate(algorithm_data.items()):
        # Sort by trust level
        trust_levels = np.array(data['trust_levels'])
        values = np.array(data['values'])
        
        sort_idx = np.argsort(trust_levels)
        trust_levels = trust_levels[sort_idx]
        values = values[sort_idx]
        
        # Plot
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        
        plt.plot(trust_levels, values, marker=marker, markersize=10, 
                linewidth=2, label=algorithm, color=color)
    
    plt.xlabel('Trust Level')
    plt.ylabel(f'Final {metric.replace("/", " - ").title()}')
    plt.title(f'{metric.replace("/", " - ").title()} vs Trust Level')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Set x-axis ticks to match the trust levels
    all_trust_levels = sorted(set(sum([data['trust_levels'] for data in algorithm_data.values()], [])))
    plt.xticks(all_trust_levels)
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt_file = os.path.join(output_dir, f"{metric.replace('/', '_')}_vs_trust.png")
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {plt_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot TensorBoard logs with Seaborn')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                      help='Specific metrics to plot (if not specified, plot all available)')
    parser.add_argument('--output', '-o', type=str, default='tensorboard_plots',
                      help='Directory to save plots')
    parser.add_argument('--smoothing', type=float, default=0.8,
                      help='Smoothing factor for curves (0-1)')
    parser.add_argument('--max-events', type=int, default=100000,
                      help='Maximum number of events to load per tag (lower is faster)')
    parser.add_argument('--no-cache', action='store_true',
                      help='Disable caching of TensorBoard data')
    parser.add_argument('--no-parallel', action='store_true',
                      help='Disable parallel processing')
    args = parser.parse_args()
    
    # Read the TensorBoard logs
    run_data = read_tensorboard_logs(
        "tensorboard_logs", 
        ["Metrics/current_speed"], 
        max_reload_size=args.max_events,
        use_cache=not args.no_cache,
        parallel=False
    )
    
    if not run_data:
        print("No valid data found in the specified directory.")
        return
    
    # Identify common metrics across all runs
    common_metrics = set()
    first = True
    
    for name, df in run_data.items():
        metrics = [col for col in df.columns if col not in ['step', 'algorithm', 'run_name']]
        if first:
            common_metrics = set(metrics)
            first = False
        else:
            common_metrics &= set(metrics)
    
    print(f"Common metrics across all runs: {common_metrics}")
    
    # If metrics specified, filter to those
    if args.metrics:
        metrics_to_plot = []
        for m in args.metrics:
            # Find metrics that contain the specified string
            matching = [metric for metric in common_metrics if m in metric]
            metrics_to_plot.extend(matching)
        metrics_to_plot = list(set(metrics_to_plot))  # Remove duplicates
    else:
        metrics_to_plot = list(common_metrics)
    
    print(f"Plotting metrics: {metrics_to_plot}")
    
    # Plot individual metrics
    for metric in metrics_to_plot:
        plot_single_metric(run_data, metric, args.output, args.smoothing)
    
    # Plot multiple metrics in a single figure
    if len(metrics_to_plot) > 1:
        plot_multiple_metrics(run_data, metrics_to_plot, args.output, args.smoothing)
    
    print("Plotting complete!")

if __name__ == "__main__":
    main() 