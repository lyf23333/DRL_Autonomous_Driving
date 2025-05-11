#!/bin/bash
#
# Installation script for TensorBoard visualization tools
# This script installs all required dependencies and sets up the environment
#

echo "Setting up TensorBoard visualization tools..."

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Create a virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " CREATE_VENV
if [[ $CREATE_VENV == "y" || $CREATE_VENV == "Y" ]]; then
    echo "Creating virtual environment..."
    python -m venv venv
    
    # Activate the virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Linux/Mac
        source venv/bin/activate
    fi
    
    echo "Virtual environment created and activated."
fi

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install tensorboard pandas numpy matplotlib seaborn

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully!"
else
    echo "Error: Failed to install dependencies."
    exit 1
fi

# Make scripts executable
chmod +x plot_tensorboard_logs.py
chmod +x plot_trust_metrics.py
chmod +x plot_evaluation_metrics.py

echo "Setup complete!"
echo "-------------------------------------"
echo "Usage examples:"
echo "python plot_tensorboard_logs.py --log-dir path/to/tensorboard/logs"
echo "python plot_trust_metrics.py --log-dir path/to/tensorboard/logs"
echo "python plot_evaluation_metrics.py --log-dir path/to/tensorboard/logs"
echo "-------------------------------------"
echo "See README_tensorboard_plots.md for more details." 