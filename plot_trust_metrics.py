#!/usr/bin/env python3
"""
Plot trust-related metrics from TensorBoard logs.
This script is a specialized wrapper around plot_tensorboard_logs.py
focusing on trust-related metrics.
"""

import os
import argparse
from plot_tensorboard_logs import read_tensorboard_logs, plot_single_metric, plot_multiple_metrics, plot_comparison_across_trust

# Define commonly used trust-related metrics
TRUST_RELATED_METRICS = [
    'rollout/ep_trust_reward_mean',
    'rollout/ep_trust_level_mean',
    'rollout/ep_safety_score_mean',
    'rollout/ep_performance_score_mean',
    'rollout/ep_trust_level_final',
    'rollout/ep_intervention_rate', 
    'rollout/ep_progress_reward_mean',
    'rollout/ep_length_mean',
    'rollout/ep_rew_mean'
]

def main():
    parser = argparse.ArgumentParser(description='Plot trust-related metrics from TensorBoard logs')
    parser.add_argument('--log-dir', type=str, required=True,
                      help='Directory containing TensorBoard logs')
    parser.add_argument('--output', '-o', type=str, default='trust_plots',
                      help='Directory to save plots')
    parser.add_argument('--smoothing', type=float, default=0.8,
                      help='Smoothing factor for curves (0-1)')
    parser.add_argument('--trust-level-plot', action='store_true',
                      help='Create trust level comparison plot')
    parser.add_argument('--custom-metrics', type=str, nargs='+', default=None,
                      help='Additional custom metrics to include')
    parser.add_argument('--max-events', type=int, default=100000,
                      help='Maximum number of events to load per tag (lower is faster)')
    parser.add_argument('--no-cache', action='store_true',
                      help='Disable caching of TensorBoard data')
    parser.add_argument('--no-parallel', action='store_true',
                      help='Disable parallel processing')
    args = parser.parse_args()
    
    # Combine default trust metrics with any custom metrics
    metrics_to_search = TRUST_RELATED_METRICS.copy()
    if args.custom_metrics:
        metrics_to_search.extend(args.custom_metrics)
    
    print(f"Searching for the following metrics: {metrics_to_search}")
    
    # Read TensorBoard logs with optimization options
    run_data = read_tensorboard_logs(
        args.log_dir, 
        metrics_to_search,
        max_reload_size=args.max_events,
        use_cache=not args.no_cache,
        parallel=not args.no_parallel
    )
    
    if not run_data:
        print("No valid data found in the specified directory.")
        return
    
    # Find which metrics are actually present in the data
    available_metrics = set()
    for name, df in run_data.items():
        available_metrics.update([col for col in df.columns 
                                 if col not in ['step', 'algorithm', 'run_name']])
    
    print(f"Available metrics: {available_metrics}")
    
    # Filter to metrics that are actually present
    metrics_to_plot = [m for m in metrics_to_search if any(m in am for am in available_metrics)]
    
    if not metrics_to_plot:
        print("None of the specified trust metrics were found in the data.")
        return
    
    print(f"Plotting the following metrics: {metrics_to_plot}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Plot individual metrics
    for metric in metrics_to_plot:
        matching_metrics = [am for am in available_metrics if metric in am]
        for m in matching_metrics:
            plot_single_metric(run_data, m, args.output, args.smoothing)
    
    # Plot multiple trust metrics in a single figure
    trust_level_metrics = [m for m in available_metrics if 'trust_level' in m]
    if trust_level_metrics:
        plot_multiple_metrics(run_data, trust_level_metrics, args.output, args.smoothing)
    
    # Plot reward metrics in a single figure
    reward_metrics = [m for m in available_metrics if 'reward' in m or 'rew' in m]
    if reward_metrics:
        plot_multiple_metrics(run_data, reward_metrics, args.output, args.smoothing)
    
    # If requested, create trust level comparison plot
    if args.trust_level_plot:
        # Extract the final trust level for each run
        trust_level_mapping = {}
        
        for name, df in run_data.items():
            trust_level_cols = [col for col in df.columns if 'trust_level' in col]
            if trust_level_cols:
                # Use the first trust level metric found
                col = trust_level_cols[0]
                # Get the final trust level
                try:
                    final_trust = df[col].iloc[-1]
                    trust_level_mapping[name] = final_trust
                except (IndexError, KeyError):
                    print(f"Could not extract final trust level for {name}")
        
        if trust_level_mapping:
            print(f"Trust level mapping: {trust_level_mapping}")
            # For each metric, create a comparison plot
            for metric in metrics_to_plot:
                matching_metrics = [am for am in available_metrics if metric in am and 'trust_level' not in am]
                for m in matching_metrics:
                    plot_comparison_across_trust(run_data, m, trust_level_mapping, args.output)
    
    print("Plotting complete!")

if __name__ == "__main__":
    main() 