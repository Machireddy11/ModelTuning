"""
Model Registry Deployment Script
Handles deployment of fine-tuned model to Google Cloud Model Registry as "ViviansLlama"
"""

import os
import logging
import yaml
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ModelRegistryDeployer:
    """Handles deployment of models to Google Cloud Model Registry"""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize Model Registry Deployer
        
        Args:
            project_id: GCP project ID
            region: GCP region
            bucket_name: GCS bucket name for model artifacts
            credentials_path: Path to service account credentials
        """
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.region = region or os.getenv('GCP_REGION', 'us-central1')
        self.bucket_name = bucket_name or os.getenv('GCP_BUCKET_NAME')
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID must be set")
        if not self.bucket_name:
            raise ValueError("GCP_BUCKET_NAME must be set")
        
        # Initialize clients
        self._initialize_clients()
        
        logger.info(f"Model Registry Deployer initialized for project: {self.project_id}")
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients"""
        credentials = None
        if self.credentials_path and os.path.exists(self.credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
        
        # Initialize AI Platform
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            credentials=credentials
        )
        
        # Initialize Storage client
        self.storage_client = storage.Client(
            project=self.project_id,
            credentials=credentials
        )
        
        logger.info("Google Cloud clients initialized")
    
    def upload_model_to_gcs(
        self,
        local_model_path: str,
        gcs_model_path: Optional[str] = None
    ) -> str:
        """
        Upload model artifacts to Google Cloud Storage
        
        Args:
            local_model_path: Local path to model directory
            gcs_model_path: GCS path (without gs:// prefix)
            
        Returns:
            Full GCS URI of uploaded model
        """
        logger.info(f"Uploading model from {local_model_path} to GCS...")
        
        if not gcs_model_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gcs_model_path = f"models/ViviansLlama/{timestamp}"
        
        bucket = self.storage_client.bucket(self.bucket_name)
        
        # Upload all files in the model directory
        local_path = Path(local_model_path)
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                blob_path = f"{gcs_model_path}/{relative_path}"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded: {blob_path}")
        
        gcs_uri = f"gs://{self.bucket_name}/{gcs_model_path}"
        logger.info(f"Model uploaded to: {gcs_uri}")
        
        return gcs_uri
    
    def register_model(
        self,
        model_artifact_uri: str,
        model_display_name: str = "ViviansLlama",
        model_description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> aiplatform.Model:
        """
        Register model in Google Cloud Model Registry
        
        Args:
            model_artifact_uri: GCS URI of model artifacts
            model_display_name: Display name for the model
            model_description: Description of the model
            labels: Labels to attach to the model
            
        Returns:
            Registered model object
        """
        logger.info(f"Registering model: {model_display_name}")
        
        if not model_description:
            model_description = (
                "LoRA fine-tuned Llama 90B model for fintech domain. "
                "Specialized in financial analysis, risk assessment, and regulatory compliance."
            )
        
        if not labels:
            labels = {
                "model_type": "llama",
                "domain": "fintech",
                "training_method": "lora",
                "version": os.getenv('MODEL_VERSION', 'v1.0.0').replace('.', '_')
            }
        
        # Upload model to Model Registry
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
            description=model_description,
            labels=labels,
        )
        
        logger.info(f"Model registered: {model.resource_name}")
        return model
    
    def create_endpoint(
        self,
        endpoint_display_name: str = "vivians-llama-endpoint",
        description: Optional[str] = None
    ) -> aiplatform.Endpoint:
        """
        Create or get an endpoint for model deployment
        
        Args:
            endpoint_display_name: Display name for the endpoint
            description: Description of the endpoint
            
        Returns:
            Endpoint object
        """
        logger.info(f"Creating/getting endpoint: {endpoint_display_name}")
        
        # Check if endpoint already exists
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )
        
        if endpoints:
            endpoint = endpoints[0]
            logger.info(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            if not description:
                description = "Endpoint for ViviansLlama - Fintech domain fine-tuned model"
            
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_display_name,
                description=description,
                project=self.project_id,
                location=self.region
            )
            logger.info(f"Created new endpoint: {endpoint.resource_name}")
        
        return endpoint
    
    def deploy_model_to_endpoint(
        self,
        model: aiplatform.Model,
        endpoint: aiplatform.Endpoint,
        machine_type: str = "n1-standard-8",
        accelerator_type: str = "NVIDIA_TESLA_V100",
        accelerator_count: int = 1,
        min_replica_count: int = 1,
        max_replica_count: int = 3,
        traffic_percentage: int = 100
    ) -> aiplatform.Endpoint:
        """
        Deploy model to an endpoint
        
        Args:
            model: Model to deploy
            endpoint: Endpoint to deploy to
            machine_type: Machine type for deployment
            accelerator_type: GPU accelerator type
            accelerator_count: Number of accelerators
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            traffic_percentage: Percentage of traffic to route to this deployment
            
        Returns:
            Updated endpoint
        """
        logger.info(f"Deploying model to endpoint...")
        
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name="ViviansLlama",
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=traffic_percentage,
        )
        
        logger.info("Model deployed successfully")
        return endpoint
    
    def full_deployment_pipeline(
        self,
        local_model_path: str,
        config_path: str = "config/training_config.yaml"
    ) -> Dict[str, Any]:
        """
        Execute full deployment pipeline
        
        Args:
            local_model_path: Local path to trained model
            config_path: Path to configuration file
            
        Returns:
            Dictionary with deployment information
        """
        logger.info("Starting full deployment pipeline...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        deployment_config = config.get('deployment', {})
        
        # Step 1: Upload model to GCS
        gcs_uri = self.upload_model_to_gcs(local_model_path)
        
        # Step 2: Register model
        model = self.register_model(
            model_artifact_uri=gcs_uri,
            model_display_name=config['model']['custom_model_name']
        )
        
        # Step 3: Create/get endpoint
        endpoint = self.create_endpoint(
            endpoint_display_name=deployment_config.get('endpoint_name', 'vivians-llama-endpoint')
        )
        
        # Step 4: Deploy model to endpoint
        endpoint = self.deploy_model_to_endpoint(
            model=model,
            endpoint=endpoint,
            machine_type=deployment_config.get('machine_type', 'n1-standard-8'),
            accelerator_type=deployment_config.get('accelerator_type', 'NVIDIA_TESLA_V100'),
            accelerator_count=deployment_config.get('accelerator_count', 1),
            min_replica_count=deployment_config.get('min_replica_count', 1),
            max_replica_count=deployment_config.get('max_replica_count', 3)
        )
        
        deployment_info = {
            'model_name': model.display_name,
            'model_resource_name': model.resource_name,
            'endpoint_name': endpoint.display_name,
            'endpoint_resource_name': endpoint.resource_name,
            'gcs_uri': gcs_uri,
            'deployment_time': datetime.now().isoformat()
        }
        
        logger.info("Full deployment pipeline completed successfully")
        logger.info(f"Deployment info: {deployment_info}")
        
        return deployment_info
    
    def test_deployed_model(
        self,
        endpoint_name: str,
        test_prompt: str = "What are the key factors in assessing credit risk?"
    ) -> Dict[str, Any]:
        """
        Test a deployed model
        
        Args:
            endpoint_name: Name of the endpoint
            test_prompt: Test prompt to send
            
        Returns:
            Prediction results
        """
        logger.info(f"Testing deployed model at endpoint: {endpoint_name}")
        
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        test_instance = {
            "prompt": test_prompt,
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = endpoint.predict(instances=[test_instance])
        
        logger.info("Model test successful")
        return {
            'prompt': test_prompt,
            'predictions': response.predictions
        }


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy model to Google Cloud Model Registry")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the deployed model after deployment"
    )
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = ModelRegistryDeployer()
    
    # Execute deployment pipeline
    deployment_info = deployer.full_deployment_pipeline(
        local_model_path=args.model_path,
        config_path=args.config
    )
    
    print("\n" + "="*50)
    print("DEPLOYMENT SUCCESSFUL")
    print("="*50)
    print(f"Model Name: {deployment_info['model_name']}")
    print(f"Model Resource: {deployment_info['model_resource_name']}")
    print(f"Endpoint Name: {deployment_info['endpoint_name']}")
    print(f"Endpoint Resource: {deployment_info['endpoint_resource_name']}")
    print(f"GCS URI: {deployment_info['gcs_uri']}")
    print(f"Deployment Time: {deployment_info['deployment_time']}")
    print("="*50 + "\n")
    
    # Test if requested
    if args.test:
        print("Testing deployed model...")
        test_results = deployer.test_deployed_model(
            endpoint_name=deployment_info['endpoint_resource_name']
        )
        print(f"\nTest Prompt: {test_results['prompt']}")
        print(f"Predictions: {test_results['predictions']}\n")


if __name__ == "__main__":
    main()

# Made with Bob
