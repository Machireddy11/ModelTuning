#!/bin/bash

# Training script for ViviansLlama
# Executes the complete training pipeline

set -e

echo "=========================================="
echo "ViviansLlama Training Pipeline"
echo "=========================================="

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment variables loaded."
else
    echo "ERROR: .env file not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated."
else
    echo "ERROR: Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Parse command line arguments
CONFIG_FILE="config/training_config.yaml"
OUTPUT_DIR="models/vivians_llama_deployment"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG_FILE] [--output OUTPUT_DIR]"
            exit 1
            ;;
    esac
done

echo "Configuration file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if training data exists
if [ ! -f "data/processed/fintech_train.jsonl" ]; then
    echo "Training data not found. Creating sample data..."
    python src/data_processing/fintech_data_processor.py --create-sample
fi

# Start training
echo "Starting LoRA fine-tuning..."
echo "This may take several hours depending on your hardware."
echo ""

python src/training/lora_finetune.py \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review training logs in logs/ directory"
echo "2. Deploy model: ./scripts/deploy.sh"
echo ""

# Made with Bob
