#!/bin/bash

# run_all_algorithms.sh
# This script runs training for all supported DRL algorithms (PPO, SAC, DDPG, DQN)
# with the same configuration settings for easy comparison.

# Set common parameters
SCENARIO="urban_traffic"  # Options: lane_switching, urban_traffic, obstacle_avoidance
TIMESTEPS=1000000           # Total timesteps for training
TOWN="Town01"              # CARLA town to use
LR=0.0003                  # Learning rate (3e-4)
LR_SCHEDULE="exponential"  # Learning rate schedule
CHECKPOINT_FREQ=10000      # Save checkpoint every N timesteps
RUN_NAME="trust_leve_rew_0.02"      # Base name for this comparison run

# Other settings
START_CARLA=true           # Whether to start CARLA server automatically
QUALITY="Low"              # Graphics quality (Low or Epic)
OFFSCREEN=true             # Run CARLA in offscreen mode (no rendering)

# Create a timestamp for the run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run an algorithm
run_algorithm() {
    pkill -f "CarlaUE4"
    ALGORITHM=$1
    
    echo "================================================"
    echo "Starting training for $ALGORITHM algorithm"
    echo "Scenario: $SCENARIO"
    echo "Timesteps: $TIMESTEPS"
    echo "Town: $TOWN"
    echo "================================================"
    
    # Build command
    CMD="python main.py"
    CMD="$CMD --algorithm $ALGORITHM"
    CMD="$CMD --scenario $SCENARIO"
    CMD="$CMD --train"
    CMD="$CMD --timesteps $TIMESTEPS"
    CMD="$CMD --town $TOWN"
    CMD="$CMD --learning-rate $LR"
    CMD="$CMD --lr-schedule $LR_SCHEDULE"
    CMD="$CMD --checkpoint-freq $CHECKPOINT_FREQ"
    CMD="$CMD --run-name ${RUN_NAME}_${ALGORITHM}"
    
    # Add conditional flags
    if $START_CARLA; then
        CMD="$CMD --start-carla"
    fi
    
    if $OFFSCREEN; then
        CMD="$CMD --offscreen"
    fi
    
    CMD="$CMD --quality $QUALITY"
    
    # Run the command
    echo "Running command: $CMD"
    eval $CMD
    
    # Wait a bit to ensure resources are released
    echo "Waiting 20 seconds before starting next algorithm..."
    pkill -f "CarlaUE4"
    sleep 20
}

# Main execution
echo "Starting sequential training of all algorithms"
echo "Timestamp: $TIMESTAMP"

# Run each algorithm in sequence
run_algorithm "sac"
run_algorithm "ddpg"
run_algorithm "dqn"

echo "All training runs completed!" 