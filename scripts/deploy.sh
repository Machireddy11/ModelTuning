#!/bin/bash

# Deployment script for ViviansLlama
# Deploys the fine-tuned model to Google Cloud Model Registry

set -e

echo "=========================================="
echo "ViviansLlama Deployment Pipeline"
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
MODEL_PATH="models/vivians_llama_deployment"
CONFIG_FILE="config/training_config.yaml"
TEST_MODEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --test)
            TEST_MODEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model-path MODEL_PATH] [--config CONFIG_FILE] [--test]"
            exit 1
            ;;
    esac
done

echo "Model path: $MODEL_PATH"
echo "Configuration file: $CONFIG_FILE"
echo "Test after deployment: $TEST_MODEL"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please train the model first using: ./scripts/train.sh"
    exit 1
fi

# Check GCP credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS not set."
    echo "Please set this environment variable to your service account key path."
    exit 1
fi

if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "ERROR: Service account key file not found at $GOOGLE_APPLICATION_CREDENTIALS"
    exit 1
fi

echo "Starting deployment to Google Cloud Model Registry..."
echo "This may take 15-30 minutes depending on model size and network speed."
echo ""

# Deploy model
if [ "$TEST_MODEL" = true ]; then
    python src/deployment/model_registry_deployment.py \
        --model-path "$MODEL_PATH" \
        --config "$CONFIG_FILE" \
        --test
else
    python src/deployment/model_registry_deployment.py \
        --model-path "$MODEL_PATH" \
        --config "$CONFIG_FILE"
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Your model 'ViviansLlama' is now deployed and ready to use."
echo ""
echo "To test the deployed model:"
echo "python src/deployment/model_registry_deployment.py --model-path $MODEL_PATH --config $CONFIG_FILE --test"
echo ""

# Made with Bob
