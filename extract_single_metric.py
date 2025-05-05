#!/usr/bin/env python3
"""
Extract a single metric from TensorBoard logs and save to CSV.
This script demonstrates the most efficient way to read TensorBoard data
when you only need a specific metric.
"""

import os
import argparse
import pandas as pd
from plot_csv_logs import read_tensorboard_logs

def extract_single_metric(log_dir, metric_name, output_file, max_events=10000):
    """
    Extract a single metric from TensorBoard logs and save to CSV.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        metric_name: Name of the metric to extract
        output_file: Path to save the output CSV
        max_events: Maximum number of events to load (lower is faster)
    """
    print(f"Extracting metric '{metric_name}' from {log_dir}")
    
    # Use optimized reading with specific metric, small event limit, and parallel processing
    run_data = read_tensorboard_logs(
        log_dir, 
        tags=[metric_name], 
        max_reload_size=max_events,
        use_cache=True,
        parallel=True
    )
    
    if not run_data:
        print("No data found.")
        return False
    
    # Check if the metric exists in any of the runs
    metric_found = False
    for name, df in run_data.items():
        if metric_name in df.columns:
            metric_found = True
            break
    
    if not metric_found:
        print(f"Metric '{metric_name}' not found in any of the runs.")
        # List available metrics
        available_metrics = set()
        for name, df in run_data.items():
            available_metrics.update([col for col in df.columns 
                                   if col not in ['step', 'algorithm', 'run_name']])
        if available_metrics:
            print("Available metrics:")
            for m in sorted(available_metrics):
                print(f"  - {m}")
        return False
    
    # Create a DataFrame for the extracted metric
    result_df = pd.DataFrame()
    
    for name, df in run_data.items():
        if metric_name in df.columns:
            # Create a DataFrame with step, value, and run name
            run_df = pd.DataFrame({
                'step': df['step'],
                'value': df[metric_name],
                'run': name,
                'algorithm': df['algorithm'].iloc[0]
            })
            
            # Append to the result DataFrame
            result_df = pd.concat([result_df, run_df], ignore_index=True)
    
    # Save to CSV
    if not result_df.empty:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        return True
    else:
        print("No data to save.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract a single metric from TensorBoard logs')
    parser.add_argument('--log-dir', type=str, required=True,
                      help='Directory containing TensorBoard logs')
    parser.add_argument('--metric', type=str, required=True,
                      help='Name of the metric to extract')
    parser.add_argument('--output', '-o', type=str, default='extracted_data.csv',
                      help='Path to save the output CSV')
    parser.add_argument('--max-events', type=int, default=10000,
                      help='Maximum number of events to load per tag (lower is faster)')
    args = parser.parse_args()
    
    extract_single_metric(args.log_dir, args.metric, args.output, args.max_events)

if __name__ == "__main__":
    main() 