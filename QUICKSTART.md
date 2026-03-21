# ViviansLlama - Quick Start Guide

Get started with LoRA fine-tuning Llama 90B for fintech in 5 steps!

## Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] Google Cloud Platform account
- [ ] GCP project with billing enabled
- [ ] AI Platform API enabled
- [ ] Cloud Storage API enabled
- [ ] Model Garden access granted
- [ ] Service account with necessary permissions
- [ ] NVIDIA GPU with 24GB+ VRAM (for training)
- [ ] Hugging Face account and token

## Step 1: Initial Setup (5 minutes)

```bash
# Run the setup script
./scripts/setup.sh

# Edit .env file with your credentials
nano .env
```

Required credentials in `.env`:
```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCP_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
MODEL_GARDEN_ENDPOINT=your-model-garden-endpoint
HF_TOKEN=your-huggingface-token
WANDB_API_KEY=your-wandb-key  # Optional
```

## Step 2: Prepare Training Data (10 minutes)

### Option A: Use Sample Data (Recommended for testing)
```bash
python src/data_processing/fintech_data_processor.py --create-sample
```

### Option B: Use Your Own Data
1. Prepare your data in JSON format:
```json
{
  "instruction": "Your task description",
  "input": "Optional context",
  "output": "Expected response",
  "category": "financial_analysis"
}
```

2. Process your data:
```bash
python src/data_processing/fintech_data_processor.py \
  --input data/raw/your_data.json \
  --output-dir data/processed \
  --format alpaca
```

## Step 3: Configure Training (5 minutes)

Edit `config/training_config.yaml` to customize:

**Key settings to update:**
- `model.model_garden_endpoint`: Your Model Garden endpoint URL
- `gcp.project_id`: Your GCP project ID
- `gcp.bucket_name`: Your GCS bucket name
- `monitoring.wandb_entity`: Your W&B username (if using)

**Optional tuning:**
- `training.num_train_epochs`: Number of training epochs (default: 3)
- `lora.r`: LoRA rank (default: 16, higher = more parameters)
- `training.learning_rate`: Learning rate (default: 2e-4)

## Step 4: Train the Model (2-8 hours)

```bash
# Start training
./scripts/train.sh

# Or with custom config
./scripts/train.sh --config config/training_config.yaml --output models/my_model
```

**Monitor training:**
- Console: Real-time progress in terminal
- W&B: https://wandb.ai/your-entity/vivians-llama-fintech
- TensorBoard: `tensorboard --logdir models/vivians_llama_fintech`

**Training time estimates:**
- Sample data (15 examples): ~30 minutes
- Small dataset (100 examples): ~2 hours
- Medium dataset (1000 examples): ~8 hours
- Large dataset (10000+ examples): 24+ hours

## Step 5: Deploy to Model Registry (15-30 minutes)

```bash
# Deploy and test
./scripts/deploy.sh --test

# Or deploy without testing
./scripts/deploy.sh
```

The deployment script will:
1. ✅ Upload model to Google Cloud Storage
2. ✅ Register as "ViviansLlama" in Model Registry
3. ✅ Create endpoint
4. ✅ Deploy model to endpoint
5. ✅ Run test inference (if --test flag used)

## Verify Deployment

### Test via Python:
```python
from src.deployment.model_registry_deployment import ModelRegistryDeployer

deployer = ModelRegistryDeployer()
result = deployer.test_deployed_model(
    endpoint_name="projects/YOUR_PROJECT/locations/us-central1/endpoints/ENDPOINT_ID",
    test_prompt="What are the key factors in assessing credit risk?"
)
print(result)
```

### Test via gcloud CLI:
```bash
gcloud ai endpoints predict ENDPOINT_ID \
  --region=us-central1 \
  --json-request=test_request.json
```

## Common Issues & Solutions

### Issue: Out of Memory during training
**Solution:**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 32
- Enable `gradient_checkpointing: true`
- Use smaller LoRA rank (r: 8)

### Issue: Model Garden connection fails
**Solution:**
- Verify GCP credentials: `gcloud auth list`
- Check API enablement: `gcloud services list --enabled`
- Confirm Model Garden access in GCP console
- Verify endpoint URL is correct

### Issue: Deployment fails
**Solution:**
- Check GCS bucket permissions
- Verify service account has required roles:
  - AI Platform Admin
  - Storage Admin
  - Vertex AI User
- Ensure sufficient quota for GPUs

### Issue: Slow training
**Solution:**
- Use multiple GPUs if available
- Enable bf16 training (already enabled by default)
- Increase batch size if memory allows
- Use faster storage (SSD) for data

## Next Steps

1. **Fine-tune hyperparameters**: Experiment with learning rate, LoRA rank, epochs
2. **Add more training data**: Collect domain-specific fintech examples
3. **Evaluate performance**: Test on held-out evaluation set
4. **Monitor in production**: Set up alerts and monitoring
5. **Iterate**: Continuously improve based on user feedback

## Useful Commands

```bash
# Check training progress
tail -f logs/training.log

# List deployed models
gcloud ai models list --region=us-central1

# List endpoints
gcloud ai endpoints list --region=us-central1

# View model details
gcloud ai models describe MODEL_ID --region=us-central1

# Delete endpoint (to save costs)
gcloud ai endpoints delete ENDPOINT_ID --region=us-central1

# Create sample data
python src/data_processing/fintech_data_processor.py --create-sample

# Process custom data
python src/data_processing/fintech_data_processor.py --input data/raw/my_data.json
```

## Cost Estimates

**Training (one-time):**
- V100 GPU: ~$2-3/hour × 2-8 hours = $4-24
- Storage: ~$0.02/GB/month

**Deployment (ongoing):**
- Endpoint with V100: ~$2-3/hour
- Storage: ~$0.02/GB/month
- Predictions: ~$0.10-0.50 per 1000 requests

**Cost optimization tips:**
- Use preemptible VMs for training (50-70% discount)
- Scale down replicas during low traffic
- Use T4 GPUs instead of V100 for inference (cheaper)
- Set up auto-shutdown for idle endpoints

## Support & Resources

- **Documentation**: See [README.md](README.md) for detailed information
- **Configuration**: Review `config/training_config.yaml` and `config/deployment_config.yaml`
- **Examples**: Check `data/examples/fintech_training_examples.json`
- **Issues**: Open an issue on GitHub

## Success Checklist

- [ ] Environment setup complete
- [ ] Training data prepared
- [ ] Configuration updated
- [ ] Model trained successfully
- [ ] Model deployed to registry
- [ ] Endpoint tested and working
- [ ] Monitoring configured
- [ ] Documentation reviewed

---

**Congratulations!** 🎉 You now have a production-ready, fine-tuned Llama 90B model specialized for fintech applications!