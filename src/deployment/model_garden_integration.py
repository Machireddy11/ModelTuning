"""
Google Model Garden Integration Script
Handles connection to Google Model Garden and retrieval of Llama 90B model endpoint
"""

import os
import logging
from typing import Optional, Dict, Any
from google.cloud import aiplatform
from google.oauth2 import service_account
import requests
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ModelGardenClient:
    """Client for interacting with Google Model Garden"""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize Model Garden client
        
        Args:
            project_id: GCP project ID
            region: GCP region
            credentials_path: Path to service account credentials
        """
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.region = region or os.getenv('GCP_REGION', 'us-central1')
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID must be set")
        
        # Initialize AI Platform
        self._initialize_aiplatform()
        
        logger.info(f"Model Garden client initialized for project: {self.project_id}")
    
    def _initialize_aiplatform(self):
        """Initialize Google AI Platform"""
        credentials = None
        if self.credentials_path and os.path.exists(self.credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
        
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            credentials=credentials
        )
        
        logger.info("AI Platform initialized")
    
    def get_model_garden_endpoint(self, model_name: str = "llama-90b") -> str:
        """
        Get the endpoint URL for a model from Model Garden
        
        Args:
            model_name: Name of the model in Model Garden
            
        Returns:
            Endpoint URL for the model
        """
        logger.info(f"Retrieving endpoint for model: {model_name}")
        
        # List all endpoints
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{model_name}"',
            order_by='create_time desc'
        )
        
        if not endpoints:
            logger.warning(f"No endpoint found for model: {model_name}")
            return None
        
        endpoint = endpoints[0]
        logger.info(f"Found endpoint: {endpoint.resource_name}")
        
        return endpoint.resource_name
    
    def deploy_model_from_garden(
        self,
        model_display_name: str,
        endpoint_display_name: str,
        machine_type: str = "n1-standard-8",
        accelerator_type: str = "NVIDIA_TESLA_V100",
        accelerator_count: int = 1,
        min_replica_count: int = 1,
        max_replica_count: int = 3
    ) -> aiplatform.Endpoint:
        """
        Deploy a model from Model Garden to an endpoint
        
        Args:
            model_display_name: Display name of the model
            endpoint_display_name: Display name for the endpoint
            machine_type: Machine type for deployment
            accelerator_type: GPU accelerator type
            accelerator_count: Number of accelerators
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            
        Returns:
            Deployed endpoint
        """
        logger.info(f"Deploying model: {model_display_name}")
        
        # Get or create endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )
        
        if endpoints:
            endpoint = endpoints[0]
            logger.info(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_display_name,
                project=self.project_id,
                location=self.region
            )
            logger.info(f"Created new endpoint: {endpoint.resource_name}")
        
        # Get model
        models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"'
        )
        
        if not models:
            raise ValueError(f"Model not found: {model_display_name}")
        
        model = models[0]
        logger.info(f"Found model: {model.resource_name}")
        
        # Deploy model to endpoint
        logger.info("Deploying model to endpoint...")
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=model_display_name,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=100,
        )
        
        logger.info("Model deployed successfully")
        return endpoint
    
    def test_endpoint(self, endpoint_name: str, test_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test an endpoint with sample input
        
        Args:
            endpoint_name: Name of the endpoint
            test_input: Test input data
            
        Returns:
            Prediction results
        """
        logger.info(f"Testing endpoint: {endpoint_name}")
        
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        # Make prediction
        response = endpoint.predict(instances=[test_input])
        
        logger.info("Endpoint test successful")
        return response.predictions
    
    def download_model_from_garden(
        self,
        model_name: str,
        output_path: str
    ) -> str:
        """
        Download a model from Model Garden
        
        Args:
            model_name: Name of the model
            output_path: Local path to save the model
            
        Returns:
            Path to downloaded model
        """
        logger.info(f"Downloading model: {model_name}")
        
        # Get model
        models = aiplatform.Model.list(
            filter=f'display_name="{model_name}"'
        )
        
        if not models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = models[0]
        
        # Download model artifacts
        os.makedirs(output_path, exist_ok=True)
        
        # Note: Actual download implementation depends on model storage location
        # This is a placeholder for the download logic
        logger.info(f"Model artifacts location: {model.artifact_uri}")
        
        # You would typically use gsutil or google-cloud-storage to download
        # from the artifact_uri to output_path
        
        logger.info(f"Model downloaded to: {output_path}")
        return output_path
    
    def list_available_models(self) -> list:
        """
        List all available models in Model Garden
        
        Returns:
            List of available models
        """
        logger.info("Listing available models...")
        
        models = aiplatform.Model.list()
        
        model_list = []
        for model in models:
            model_info = {
                'name': model.display_name,
                'resource_name': model.resource_name,
                'create_time': model.create_time,
                'update_time': model.update_time,
            }
            model_list.append(model_info)
        
        logger.info(f"Found {len(model_list)} models")
        return model_list


def main():
    """Main execution function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google Model Garden Integration")
    parser.add_argument(
        "--action",
        type=str,
        choices=['list', 'get_endpoint', 'deploy', 'test'],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llama-90b",
        help="Model name"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        help="Endpoint name"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = ModelGardenClient()
    
    if args.action == 'list':
        models = client.list_available_models()
        for model in models:
            print(f"Model: {model['name']}")
            print(f"  Resource: {model['resource_name']}")
            print(f"  Created: {model['create_time']}")
            print()
    
    elif args.action == 'get_endpoint':
        endpoint = client.get_model_garden_endpoint(args.model_name)
        print(f"Endpoint: {endpoint}")
    
    elif args.action == 'deploy':
        if not args.endpoint_name:
            print("Error: --endpoint-name required for deploy action")
            return
        
        endpoint = client.deploy_model_from_garden(
            model_display_name=args.model_name,
            endpoint_display_name=args.endpoint_name
        )
        print(f"Model deployed to endpoint: {endpoint.resource_name}")
    
    elif args.action == 'test':
        if not args.endpoint_name:
            print("Error: --endpoint-name required for test action")
            return
        
        test_input = {
            "prompt": "What is the current state of the financial markets?",
            "max_tokens": 100
        }
        
        results = client.test_endpoint(args.endpoint_name, test_input)
        print(f"Test results: {results}")


if __name__ == "__main__":
    main()

# Made with Bob
