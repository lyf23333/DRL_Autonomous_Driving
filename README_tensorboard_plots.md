# TensorBoard Log Visualization Tools

This directory contains tools for extracting and visualizing data from TensorBoard logs, with a focus on creating publication-quality plots using Seaborn.

## Requirements

Install the required packages:

```bash
pip install tensorboard pandas numpy matplotlib seaborn
```

## Scripts

### `plot_tensorboard_logs.py`

A general-purpose script for reading TensorBoard logs and creating visualizations.

#### Basic Usage

```bash
python plot_tensorboard_logs.py --log-dir /path/to/tensorboard/logs --output plot_output
```

#### Options

- `--log-dir`: Directory containing TensorBoard logs (required)
- `--metrics`: Specific metrics to plot (optional, plots all available metrics if not specified)
- `--output` or `-o`: Directory to save plots (default: 'tensorboard_plots')
- `--smoothing`: Smoothing factor for curves, between 0-1 (default: 0.8)
- `--max-events`: Maximum number of events to load per tag (default: 100000, lower is faster)
- `--no-cache`: Disable caching of TensorBoard data
- `--no-parallel`: Disable parallel processing

#### Performance Optimization

This script includes several features to improve performance when dealing with large TensorBoard logs:

1. **Caching**: Results are cached to speed up subsequent runs with the same data
2. **Parallel Processing**: Multiple log directories are processed in parallel
3. **Size Limits**: You can limit the number of events loaded per tag
4. **Metric Filtering**: Only load the metrics you need

#### Examples

Plot all metrics from logs:
```bash
python plot_tensorboard_logs.py --log-dir runs/
```

Plot only specific metrics:
```bash
python plot_tensorboard_logs.py --log-dir runs/ --metrics reward trust_level
```

Change smoothing factor:
```bash
python plot_tensorboard_logs.py --log-dir runs/ --smoothing 0.5
```

Optimize for speed with large logs:
```bash
python plot_tensorboard_logs.py --log-dir runs/ --metrics reward --max-events 10000
```

Disable caching and parallel processing:
```bash
python plot_tensorboard_logs.py --log-dir runs/ --no-cache --no-parallel
```

### `plot_trust_metrics.py`

A specialized script for visualizing trust-related metrics from TensorBoard logs.

#### Basic Usage

```bash
python plot_trust_metrics.py --log-dir /path/to/tensorboard/logs
```

#### Options

- `--log-dir`: Directory containing TensorBoard logs (required)
- `--output` or `-o`: Directory to save plots (default: 'trust_plots')
- `--smoothing`: Smoothing factor for curves, between 0-1 (default: 0.8)
- `--trust-level-plot`: Generate plots comparing metrics against trust levels
- `--custom-metrics`: Additional custom metrics to include beyond the default trust-related ones
- `--max-events`: Maximum number of events to load per tag (default: 100000, lower is faster)
- `--no-cache`: Disable caching of TensorBoard data
- `--no-parallel`: Disable parallel processing

#### Examples

Generate basic trust metric plots:
```bash
python plot_trust_metrics.py --log-dir runs/
```

Generate trust level comparison plots:
```bash
python plot_trust_metrics.py --log-dir runs/ --trust-level-plot
```

Include custom metrics:
```bash
python plot_trust_metrics.py --log-dir runs/ --custom-metrics rollout/ep_collision_count rewards/path_following
```

Generate plots quickly from large logs:
```bash
python plot_trust_metrics.py --log-dir runs/ --max-events 5000
```

### `plot_evaluation_metrics.py`

A specialized script for generating evaluation report plots and tables.

#### Basic Usage

```bash
python plot_evaluation_metrics.py --log-dir /path/to/tensorboard/logs --output evaluation_output
```

#### Options

Similar to the other scripts, plus:
- `--table-only`: Only generate the metrics table, no plots

## Generated Plots

The scripts generate several types of visualizations:

1. **Individual metric plots**: Single-metric visualizations showing the metric's trend over training steps
2. **Multi-metric plots**: Combined visualizations with multiple metrics in subplots
3. **Trust level comparison plots**: Visualizations comparing metrics against trust levels (requires `--trust-level-plot` option)

## Performance Tips

For large TensorBoard logs, here are some tips to improve performance:

1. **Specify metrics**: Only load the metrics you need by using the `--metrics` parameter
2. **Limit events**: Use `--max-events` with a lower value (e.g., 5000-10000) to sample fewer points
3. **Use caching**: The first run will be slower, but subsequent runs will be much faster
4. **Parallel processing**: This is enabled by default and speeds up processing when you have multiple log directories

## How the Scripts Work

1. The scripts scan the specified directory for TensorBoard log files
2. Event files are parsed using TensorBoard's `EventAccumulator`
3. Data is extracted and organized into pandas DataFrames
4. Visualizations are created using Seaborn and Matplotlib
5. Plots are saved to the specified output directory

## Customizing the Plots

You can customize the plots by modifying the source code:

- Change plot styles by modifying the `sns.set_context` and `plt.style.use` calls
- Adjust colors by modifying the `COLORS` list
- Change markers by modifying the `MARKERS` list
- Adjust smoothing behavior in the `smooth_data` function

## For Developers

The code is modular, with separate functions for:
- Reading TensorBoard logs
- Smoothing data
- Plotting individual metrics
- Plotting multiple metrics
- Plotting comparisons across trust levels

You can import these functions into your own scripts for custom visualizations.

## Troubleshooting

- If no data is found, check that the log directory contains valid TensorBoard event files
- If specific metrics are not appearing, check available metrics in the console output
- For large log files, try reducing the `--max-events` value to improve performance
- If you experience memory issues, try disabling parallel processing with `--no-parallel`
- If the cache becomes outdated, use `--no-cache` to force a fresh read 