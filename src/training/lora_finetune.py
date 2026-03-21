"""
LoRA Fine-tuning Script for Llama 90B Model - Fintech Domain
This script handles the fine-tuning of Llama 90B using LoRA (Low-Rank Adaptation)
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import load_dataset
import wandb
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LlamaLoRATrainer:
    """Main class for LoRA fine-tuning of Llama 90B model"""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """
        Initialize the trainer with configuration
        
        Args:
            config_path: Path to the training configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def setup_model_and_tokenizer(self):
        """Setup the model with quantization and LoRA configuration"""
        logger.info("Setting up model and tokenizer...")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
        )
        
        # Load tokenizer
        model_name = self.config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=os.getenv('HF_TOKEN')
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=os.getenv('HF_TOKEN')
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type'],
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
        
    def load_and_prepare_data(self):
        """Load and prepare the fintech training data"""
        logger.info("Loading training data...")
        
        train_path = self.config['data']['train_data_path']
        eval_path = self.config['data']['eval_data_path']
        
        # Load datasets
        train_dataset = load_dataset('json', data_files=train_path, split='train')
        eval_dataset = load_dataset('json', data_files=eval_path, split='train')
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.config['data']['dataset_text_field']],
                truncation=True,
                max_length=self.config['data']['max_seq_length'],
                padding="max_length",
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments"""
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_ratio=training_config['warmup_ratio'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            save_total_limit=training_config['save_total_limit'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
            max_grad_norm=training_config['max_grad_norm'],
            max_steps=training_config['max_steps'],
            group_by_length=training_config['group_by_length'],
            report_to=training_config['report_to'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            push_to_hub=False,
        )
    
    def train(self):
        """Execute the training process"""
        logger.info("Starting training process...")
        
        # Initialize wandb
        if 'wandb' in self.config['training']['report_to']:
            wandb.init(
                project=self.config['monitoring']['wandb_project'],
                entity=self.config['monitoring'].get('wandb_entity'),
                name=self.config['monitoring']['experiment_name'],
                config=self.config,
            )
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load data
        train_dataset, eval_dataset = self.load_and_prepare_data()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info("Training complete!")
        
        if 'wandb' in self.config['training']['report_to']:
            wandb.finish()
    
    def save_for_deployment(self, output_path: str):
        """
        Save the fine-tuned model for deployment
        
        Args:
            output_path: Path to save the deployment-ready model
        """
        logger.info(f"Saving model for deployment to {output_path}")
        
        # Merge LoRA weights with base model
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info("Model saved for deployment")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Llama 90B")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/vivians_llama_deployment",
        help="Path to save deployment-ready model"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LlamaLoRATrainer(config_path=args.config)
    
    # Train model
    trainer.train()
    
    # Save for deployment
    trainer.save_for_deployment(args.output)
    
    logger.info("All operations completed successfully!")


if __name__ == "__main__":
    main()

# Made with Bob
