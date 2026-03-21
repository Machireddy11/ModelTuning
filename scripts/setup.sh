#!/bin/bash

# Setup script for ViviansLlama project
# This script sets up the environment and prepares for training

set -e

echo "=========================================="
echo "ViviansLlama Project Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created. Please edit it with your credentials."
else
    echo ".env file already exists."
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{raw,processed,examples}
mkdir -p models
mkdir -p logs
mkdir -p notebooks

# Check for Google Cloud credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "WARNING: GOOGLE_APPLICATION_CREDENTIALS not set."
    echo "Please set this environment variable to your service account key path."
fi

# Check for required environment variables
echo "Checking environment variables..."
if [ -f ".env" ]; then
    source .env
    
    required_vars=("GCP_PROJECT_ID" "GCP_BUCKET_NAME")
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "WARNING: The following required variables are not set in .env:"
        printf '%s\n' "${missing_vars[@]}"
    else
        echo "All required environment variables are set."
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials"
echo "2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable"
echo "3. Prepare your training data or use sample data"
echo "4. Run: python src/data_processing/fintech_data_processor.py --create-sample"
echo "5. Start training: python src/training/lora_finetune.py"
echo ""
echo "For more information, see README.md"
echo ""

# Made with Bob
