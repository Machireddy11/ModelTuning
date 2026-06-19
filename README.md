# Llama - LoRA Fine-tuned Llama 90B for Fintech Domain

A comprehensive project for fine-tuning Meta's Llama 90B model using LoRA (Low-Rank Adaptation) for fintech domain applications, with integration to Google Model Garden and deployment to Google Cloud Model Registry.

## 🎯 Project Overview

This project provides an end-to-end pipeline for:
- LoRA fine-tuning of Llama 90B model on fintech-specific data
- Integration with Google Model Garden for base model access
- Deployment to Google Cloud Model Registry as "ViviansLlama"
- Automated endpoint creation and model serving

## 📋 Features

- **LoRA Fine-tuning**: Efficient fine-tuning using 4-bit quantization and LoRA adapters
- **Fintech Specialization**: Trained on financial analysis, risk assessment, regulatory compliance, and market analysis
- **Google Cloud Integration**: Seamless integration with GCP services
- **Model Registry**: Automated deployment to custom model registry
- **Production Ready**: Includes monitoring, logging, and testing utilities

## 🏗️ Project Structure

```
.
├── config/                      # Configuration files
│   └── training_config.yaml     # Training and deployment configuration
├── data/                        # Data directory
│   ├── raw/                     # Raw training data
│   ├── processed/               # Processed training data
│   └── examples/                # Example datasets
├── src/                         # Source code
│   ├── training/                # Training scripts
│   │   └── lora_finetune.py    # Main LoRA fine-tuning script
│   ├── deployment/              # Deployment scripts
│   │   ├── model_garden_integration.py    # Google Model Garden integration
│   │   └── model_registry_deployment.py   # Model registry deployment
│   ├── data_processing/         # Data processing utilities
│   │   └── fintech_data_processor.py      # Fintech data processor
│   └── utils/                   # Utility functions
├── models/                      # Saved models
├── logs/                        # Training logs
├── notebooks/                   # Jupyter notebooks
├── scripts/                     # Helper scripts
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Google Cloud Platform account with:
  - AI Platform API enabled
  - Cloud Storage API enabled
  - Model Garden access
- NVIDIA GPU with at least 24GB VRAM (for training)
- Hugging Face account (for model access)

### Installation

1. **Clone the repository**
```bash
cd /path/to/your/workspace
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_REGION`: GCP region (e.g., us-central1)
- `GCP_BUCKET_NAME`: GCS bucket for model artifacts
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key
- `MODEL_GARDEN_ENDPOINT`: Google Model Garden endpoint URL
- `HF_TOKEN`: Hugging Face token (for Llama access)
- `WANDB_API_KEY`: Weights & Biases API key (optional)

### Configuration

Edit `config/training_config.yaml` to customize:
- Model parameters
- LoRA configuration
- Training hyperparameters
- Deployment settings
- GCP configuration

## 📊 Data Preparation

### Create Sample Data

Generate sample fintech training data:
```bash
python src/data_processing/fintech_data_processor.py --create-sample
```

### Process Your Own Data

Prepare your data in JSON or JSONL format with the following structure:
```json
{
  "instruction": "Your instruction here",
  "input": "Optional context",
  "output": "Expected response",
  "category": "financial_analysis"
}
```

Process your data:
```bash
python src/data_processing/fintech_data_processor.py \
  --input data/raw/your_data.json \
  --output-dir data/processed \
  --format alpaca \
  --eval-ratio 0.1
```

## 🎓 Training

### Start LoRA Fine-tuning

```bash
python src/training/lora_finetune.py \
  --config config/training_config.yaml \
  --output models/vivians_llama_deployment
```

The training script will:
1. Load the Llama 90B model from Google Model Garden
2. Apply 4-bit quantization for memory efficiency
3. Configure and apply LoRA adapters
4. Train on fintech data
5. Save the fine-tuned model

### Monitor Training

Training metrics are logged to:
- **Weights & Biases**: Real-time training metrics
- **TensorBoard**: Local metric visualization
- **Console logs**: Progress and status updates

View TensorBoard:
```bash
tensorboard --logdir models/vivians_llama_fintech
```

## 🚢 Deployment

### Google Model Garden Integration

List available models:
```bash
python src/deployment/model_garden_integration.py \
  --action list
```

Get model endpoint:
```bash
python src/deployment/model_garden_integration.py \
  --action get_endpoint \
  --model-name llama-90b
```

### Deploy to Model Registry

Deploy the fine-tuned model as "ViviansLlama":
```bash
python src/deployment/model_registry_deployment.py \
  --model-path models/vivians_llama_deployment \
  --config config/training_config.yaml \
  --test
```

This will:
1. Upload model artifacts to Google Cloud Storage
2. Register model in Model Registry as "ViviansLlama"
3. Create or update endpoint
4. Deploy model to endpoint
5. Run test inference (if --test flag is used)

## 🧪 Testing

### Test Deployed Model

```python
from src.deployment.model_registry_deployment import ModelRegistryDeployer

deployer = ModelRegistryDeployer()
result = deployer.test_deployed_model(
    endpoint_name="projects/YOUR_PROJECT/locations/us-central1/endpoints/ENDPOINT_ID",
    test_prompt="What are the key factors in credit risk assessment?"
)
print(result)
```

## 📈 Model Performance

The fine-tuned model specializes in:
- **Financial Analysis**: Market trends, financial statements, investment strategies
- **Risk Assessment**: Credit risk, market risk, operational risk evaluation
- **Regulatory Compliance**: Understanding of financial regulations (Dodd-Frank, Basel III, etc.)
- **Fraud Detection**: Identifying suspicious transactions and patterns
- **Investment Advice**: Portfolio management and investment recommendations

## 🔧 Advanced Configuration

### LoRA Parameters

Adjust in `config/training_config.yaml`:
- `r`: LoRA rank (default: 16) - Higher values = more parameters
- `lora_alpha`: LoRA alpha (default: 32) - Scaling factor
- `target_modules`: Which model layers to apply LoRA to
- `lora_dropout`: Dropout rate for LoRA layers

### Training Parameters

Key hyperparameters:
- `learning_rate`: 2e-4 (recommended for LoRA)
- `num_train_epochs`: 3 (adjust based on dataset size)
- `per_device_train_batch_size`: 1 (increase if GPU memory allows)
- `gradient_accumulation_steps`: 16 (effective batch size = 16)

### Deployment Configuration

Customize deployment:
- `machine_type`: VM instance type
- `accelerator_type`: GPU type (V100, A100, etc.)
- `min_replica_count`: Minimum instances
- `max_replica_count`: Maximum instances for autoscaling

## 📝 Best Practices

1. **Data Quality**: Ensure high-quality, diverse fintech examples
2. **Validation**: Always validate on held-out test set
3. **Monitoring**: Track training metrics and model performance
4. **Version Control**: Tag model versions in registry
5. **Security**: Never commit credentials or API keys
6. **Cost Management**: Monitor GCP costs, especially for GPU usage

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Use smaller LoRA rank

**Slow Training**
- Increase batch size if memory allows
- Use multiple GPUs with `device_map="auto"`
- Enable mixed precision training (bf16)

**Model Garden Connection Issues**
- Verify GCP credentials
- Check API enablement
- Confirm Model Garden access permissions

**Deployment Failures**
- Verify GCS bucket permissions
- Check endpoint quotas
- Ensure sufficient compute resources

## 📚 Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Llama 2 Model Card](https://huggingface.co/meta-llama)
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with:
- Llama 2 Community License Agreement
- Google Cloud Terms of Service
- Applicable financial regulations

## 👥 Authors

- **Vivian** - Initial work and project setup

## 🙏 Acknowledgments

- Meta AI for Llama 2 model
- Hugging Face for PEFT library
- Google Cloud for infrastructure
- The open-source ML community

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Note**: This project requires access to Google Model Garden and appropriate GCP permissions. Ensure all prerequisites are met before starting.
