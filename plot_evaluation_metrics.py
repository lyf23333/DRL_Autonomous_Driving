#!/usr/bin/env python3
"""
Generate evaluation plots for the quantitative evaluation section.
This script creates standardized plots for comparing different algorithms
across trust levels and metrics for the evaluation report.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from plot_csv_logs import read_tensorboard_logs, plot_single_metric, plot_multiple_metrics, plot_comparison_across_trust

# Key metrics for evaluation
EVALUATION_METRICS = {
    'performance': [
        'rollout/ep_rew_mean',
        'rollout/ep_length_mean', 
        'rollout/ep_performance_score_mean'
    ],
    'safety': [
        'rollout/ep_safety_score_mean',
        'rollout/ep_collision_count_mean',
        'rollout/ep_intervention_rate'
    ],
    'trust': [
        'rollout/ep_trust_level_mean',
        'rollout/ep_trust_level_final',
        'rollout/ep_trust_reward_mean'
    ],
    'progress': [
        'rollout/ep_progress_reward_mean',
        'rollout/ep_progress_percent_mean',
        'rollout/ep_avg_speed_mean'
    ]
}

def create_evaluation_report_plots(log_dir, output_dir, smoothing=0.8, max_events=100000, use_cache=True, parallel=True):
    """
    Create standardized plots for the evaluation report.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save the plots
        smoothing: Smoothing factor for the curves
        max_events: Maximum number of events to load per tag
        use_cache: Whether to use caching for faster loading
        parallel: Whether to use parallel processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different plot types
    performance_dir = os.path.join(output_dir, "performance")
    comparison_dir = os.path.join(output_dir, "comparison")
    combined_dir = os.path.join(output_dir, "combined")
    
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # Collect all metrics to search for
    all_metrics = []
    for category in EVALUATION_METRICS.values():
        all_metrics.extend(category)
    
    # Read the TensorBoard logs with optimization options
    print(f"Reading TensorBoard logs from {log_dir}...")
    run_data = read_tensorboard_logs(
        log_dir, 
        all_metrics, 
        max_reload_size=max_events, 
        use_cache=use_cache, 
        parallel=parallel
    )
    
    if not run_data:
        print("No valid data found in the specified directory.")
        return
    
    # Find available metrics in the data
    available_metrics = set()
    for name, df in run_data.items():
        available_metrics.update([col for col in df.columns 
                                if col not in ['step', 'algorithm', 'run_name']])
    
    print(f"Available metrics: {available_metrics}")
    
    # Generate plots for each category
    for category, metrics in EVALUATION_METRICS.items():
        print(f"\nProcessing {category} metrics...")
        
        # Filter to metrics that are actually present
        category_metrics = [m for m in metrics if any(m in am for am in available_metrics)]
        
        if not category_metrics:
            print(f"No {category} metrics found in the data.")
            continue
        
        # Plot individual metrics
        for metric in category_metrics:
            matching_metrics = [am for am in available_metrics if metric in am]
            for m in matching_metrics:
                plot_single_metric(run_data, m, performance_dir, smoothing)
        
        # Plot all metrics in this category together
        category_available_metrics = []
        for metric in category_metrics:
            category_available_metrics.extend([am for am in available_metrics if metric in am])
        
        if category_available_metrics:
            print(f"Creating combined plot for {category} metrics...")
            plot_multiple_metrics(run_data, category_available_metrics, 
                                combined_dir, smoothing)
    
    # Extract trust levels for comparison plots
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
        print(f"\nCreating trust level comparison plots...")
        print(f"Trust level mapping: {trust_level_mapping}")
        
        # Create comparison plots for key metrics
        key_comparison_metrics = [
            'rollout/ep_rew_mean',
            'rollout/ep_performance_score_mean',
            'rollout/ep_safety_score_mean',
            'rollout/ep_progress_percent_mean'
        ]
        
        for metric in key_comparison_metrics:
            matching_metrics = [am for am in available_metrics if metric in am]
            for m in matching_metrics:
                plot_comparison_across_trust(run_data, m, trust_level_mapping, comparison_dir)
    
    print("\nPlotting complete! Files saved to:")
    print(f"- Performance plots: {performance_dir}")
    print(f"- Comparison plots: {comparison_dir}")
    print(f"- Combined plots: {combined_dir}")

def create_final_report_table(log_dir, output_file, max_events=100000, use_cache=True, parallel=True):
    """
    Create a CSV table with final metrics for the evaluation report.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_file: Path to save the CSV file
        max_events: Maximum number of events to load per tag
        use_cache: Whether to use caching for faster loading
        parallel: Whether to use parallel processing
    """
    # Collect all metrics to search for
    all_metrics = []
    for category in EVALUATION_METRICS.values():
        all_metrics.extend(category)
    
    # Read the TensorBoard logs with optimization options
    run_data = read_tensorboard_logs(
        log_dir, 
        all_metrics, 
        max_reload_size=max_events, 
        use_cache=use_cache, 
        parallel=parallel
    )
    
    if not run_data:
        print("No valid data found in the specified directory.")
        return
    
    # Create a table with the final metrics
    table_data = []
    
    for name, df in run_data.items():
        algorithm = df['algorithm'].iloc[0]
        run_name = df['run_name'].iloc[0]
        
        # Extract relevant metrics
        row_data = {
            'Algorithm': algorithm,
            'Run Name': run_name
        }
        
        # Add trust level if available
        trust_level_cols = [col for col in df.columns if 'trust_level' in col]
        if trust_level_cols:
            col = trust_level_cols[0]
            try:
                row_data['Trust Level'] = df[col].iloc[-1]
            except (IndexError, KeyError):
                row_data['Trust Level'] = None
        
        # Add final metrics
        for metric in all_metrics:
            matching_metrics = [col for col in df.columns if metric in col]
            for col in matching_metrics:
                try:
                    # Use a more readable name in the table
                    readable_name = col.split('/')[-1].replace('ep_', '').replace('_mean', '').replace('_', ' ').title()
                    row_data[readable_name] = df[col].iloc[-1]
                except (IndexError, KeyError):
                    pass
        
        table_data.append(row_data)
    
    # Convert to DataFrame and save
    if table_data:
        df = pd.DataFrame(table_data)
        df.to_csv(output_file, index=False)
        print(f"Final metrics table saved to {output_file}")
    else:
        print("No data available for the table.")

def main():
    parser = argparse.ArgumentParser(description='Generate evaluation plots and tables for the report')
    parser.add_argument('--log-dir', type=str, required=True,
                      help='Directory containing TensorBoard logs')
    parser.add_argument('--output', '-o', type=str, default='evaluation_plots',
                      help='Directory to save plots and tables')
    parser.add_argument('--smoothing', type=float, default=0.8,
                      help='Smoothing factor for curves (0-1)')
    parser.add_argument('--max-events', type=int, default=100000,
                      help='Maximum number of events to load per tag (lower is faster)')
    parser.add_argument('--no-cache', action='store_true',
                      help='Disable caching of TensorBoard data')
    parser.add_argument('--no-parallel', action='store_true',
                      help='Disable parallel processing')
    parser.add_argument('--table-only', action='store_true',
                      help='Only generate the metrics table, no plots')
    args = parser.parse_args()
    
    if not args.table_only:
        create_evaluation_report_plots(
            args.log_dir, 
            args.output, 
            args.smoothing,
            args.max_events,
            not args.no_cache,
            not args.no_parallel
        )
    
    # Create metrics table
    table_file = os.path.join(args.output, "final_metrics.csv")
    create_final_report_table(
        args.log_dir, 
        table_file,
        args.max_events,
        not args.no_cache,
        not args.no_parallel
    )

if __name__ == "__main__":
    main() 