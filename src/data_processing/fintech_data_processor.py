"""
Fintech Data Processor
Handles preprocessing and formatting of fintech domain data for model training
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FintechDataProcessor:
    """Process and format fintech data for LoRA fine-tuning"""
    
    def __init__(
        self,
        instruction_format: str = "alpaca",
        max_length: int = 2048
    ):
        """
        Initialize data processor
        
        Args:
            instruction_format: Format for instructions (alpaca, chatml, vicuna)
            max_length: Maximum sequence length
        """
        self.instruction_format = instruction_format
        self.max_length = max_length
        
        logger.info(f"Data processor initialized with format: {instruction_format}")
    
    def format_instruction_alpaca(
        self,
        instruction: str,
        input_text: str = "",
        output: str = ""
    ) -> str:
        """
        Format data in Alpaca instruction format
        
        Args:
            instruction: The instruction/task
            input_text: Optional input context
            output: Expected output/response
            
        Returns:
            Formatted text string
        """
        if input_text:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        
        return prompt
    
    def format_instruction_chatml(
        self,
        instruction: str,
        input_text: str = "",
        output: str = ""
    ) -> str:
        """
        Format data in ChatML format
        
        Args:
            instruction: The instruction/task
            input_text: Optional input context
            output: Expected output/response
            
        Returns:
            Formatted text string
        """
        if input_text:
            full_instruction = f"{instruction}\n\n{input_text}"
        else:
            full_instruction = instruction
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant specialized in financial analysis and fintech domain expertise.<|im_end|>
<|im_start|>user
{full_instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        
        return prompt
    
    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        output: str = ""
    ) -> str:
        """
        Format instruction based on configured format
        
        Args:
            instruction: The instruction/task
            input_text: Optional input context
            output: Expected output/response
            
        Returns:
            Formatted text string
        """
        if self.instruction_format == "alpaca":
            return self.format_instruction_alpaca(instruction, input_text, output)
        elif self.instruction_format == "chatml":
            return self.format_instruction_chatml(instruction, input_text, output)
        else:
            # Default simple format
            if input_text:
                return f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
            return f"Instruction: {instruction}\nResponse: {output}"
    
    def process_fintech_examples(
        self,
        examples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Process a list of fintech examples
        
        Args:
            examples: List of dictionaries with instruction, input, output keys
            
        Returns:
            List of processed examples with formatted text
        """
        processed = []
        
        for example in examples:
            formatted_text = self.format_instruction(
                instruction=example.get('instruction', ''),
                input_text=example.get('input', ''),
                output=example.get('output', '')
            )
            
            processed.append({
                'text': formatted_text,
                'instruction': example.get('instruction', ''),
                'input': example.get('input', ''),
                'output': example.get('output', ''),
                'category': example.get('category', 'general')
            })
        
        return processed
    
    def load_from_json(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load examples from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of examples
        """
        logger.info(f"Loading data from {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'examples' in data:
            data = data['examples']
        
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def load_from_jsonl(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load examples from JSONL file
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of examples
        """
        logger.info(f"Loading data from {file_path}")
        
        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(examples)} examples")
        return examples
    
    def save_to_jsonl(
        self,
        examples: List[Dict[str, str]],
        output_path: str
    ):
        """
        Save processed examples to JSONL file
        
        Args:
            examples: List of processed examples
            output_path: Path to save JSONL file
        """
        logger.info(f"Saving {len(examples)} examples to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Data saved to {output_path}")
    
    def create_train_eval_split(
        self,
        examples: List[Dict[str, str]],
        eval_ratio: float = 0.1,
        seed: int = 42
    ) -> tuple:
        """
        Split data into train and evaluation sets
        
        Args:
            examples: List of examples
            eval_ratio: Ratio of data to use for evaluation
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_examples, eval_examples)
        """
        import random
        random.seed(seed)
        
        # Shuffle examples
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * (1 - eval_ratio))
        train_examples = shuffled[:split_idx]
        eval_examples = shuffled[split_idx:]
        
        logger.info(f"Split data: {len(train_examples)} train, {len(eval_examples)} eval")
        
        return train_examples, eval_examples
    
    def process_and_save(
        self,
        input_path: str,
        output_dir: str = "data/processed",
        eval_ratio: float = 0.1
    ):
        """
        Complete processing pipeline: load, process, split, and save
        
        Args:
            input_path: Path to input data file
            output_dir: Directory to save processed data
            eval_ratio: Ratio for evaluation split
        """
        logger.info("Starting data processing pipeline...")
        
        # Load data
        if input_path.endswith('.jsonl'):
            raw_examples = self.load_from_jsonl(input_path)
        else:
            raw_examples = self.load_from_json(input_path)
        
        # Process examples
        processed_examples = self.process_fintech_examples(raw_examples)
        
        # Split into train/eval
        train_examples, eval_examples = self.create_train_eval_split(
            processed_examples,
            eval_ratio=eval_ratio
        )
        
        # Save processed data
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, "fintech_train.jsonl")
        eval_path = os.path.join(output_dir, "fintech_eval.jsonl")
        
        self.save_to_jsonl(train_examples, train_path)
        self.save_to_jsonl(eval_examples, eval_path)
        
        logger.info("Data processing pipeline completed successfully")
        
        return {
            'train_path': train_path,
            'eval_path': eval_path,
            'train_count': len(train_examples),
            'eval_count': len(eval_examples)
        }


def create_sample_fintech_data() -> List[Dict[str, str]]:
    """
    Create sample fintech training data
    
    Returns:
        List of sample examples
    """
    return [
        {
            "instruction": "Analyze the credit risk of a loan applicant",
            "input": "Applicant has a credit score of 720, annual income of $85,000, debt-to-income ratio of 28%, and 5 years of employment history.",
            "output": "Based on the provided information, this applicant presents a moderate-to-low credit risk profile. The credit score of 720 is considered good, indicating responsible credit management. The debt-to-income ratio of 28% is within acceptable limits (typically below 36%). The stable employment history of 5 years demonstrates income stability. However, I would recommend verifying the applicant's payment history, current outstanding debts, and the purpose of the loan to make a comprehensive risk assessment.",
            "category": "risk_assessment"
        },
        {
            "instruction": "Explain the concept of diversification in investment portfolios",
            "input": "",
            "output": "Diversification is a risk management strategy that involves spreading investments across various financial instruments, industries, and other categories to reduce exposure to any single asset or risk. The principle behind diversification is that a portfolio of different kinds of investments will, on average, yield higher returns and pose a lower risk than any individual investment. This is because different assets often perform differently under the same market conditions. For example, when stocks decline, bonds might hold steady or increase in value. A well-diversified portfolio typically includes a mix of stocks, bonds, real estate, and other asset classes, balanced according to the investor's risk tolerance and investment goals.",
            "category": "investment_advice"
        },
        {
            "instruction": "Identify potential fraud indicators in a transaction",
            "input": "Transaction details: $15,000 wire transfer to an overseas account, initiated at 2 AM, from a customer who typically makes transactions under $500, and the destination country is flagged as high-risk.",
            "output": "This transaction exhibits several red flags for potential fraud: 1) Unusual transaction amount - significantly higher than the customer's typical pattern, 2) Suspicious timing - initiated during unusual hours (2 AM), 3) High-risk destination - transfer to a country with elevated fraud risk, 4) Transaction type - wire transfers, especially international ones, are commonly used in fraud schemes. I recommend immediate verification through: contacting the customer via verified phone number, implementing additional authentication measures, temporarily holding the transaction for review, and checking for any recent account compromise indicators such as password changes or new device logins.",
            "category": "fraud_detection"
        },
        {
            "instruction": "Summarize key requirements of the Dodd-Frank Act",
            "input": "",
            "output": "The Dodd-Frank Wall Street Reform and Consumer Protection Act, enacted in 2010, introduced comprehensive financial regulatory reforms. Key requirements include: 1) Volcker Rule - restricts banks from proprietary trading and limits investments in hedge funds and private equity, 2) Enhanced capital requirements - banks must maintain higher capital reserves, 3) Stress testing - large financial institutions must undergo regular stress tests, 4) Consumer Financial Protection Bureau (CFPB) - created to protect consumers from predatory lending practices, 5) Derivatives regulation - requires standardized derivatives to be traded on exchanges and cleared through clearinghouses, 6) Resolution authority - provides framework for orderly liquidation of failing financial institutions, 7) Whistleblower provisions - protects and incentivizes reporting of securities violations.",
            "category": "regulatory_compliance"
        },
        {
            "instruction": "Calculate the compound annual growth rate (CAGR)",
            "input": "Initial investment: $10,000, Final value: $18,500, Time period: 5 years",
            "output": "To calculate the Compound Annual Growth Rate (CAGR), we use the formula: CAGR = (Ending Value / Beginning Value)^(1/Number of Years) - 1. Plugging in the values: CAGR = ($18,500 / $10,000)^(1/5) - 1 = (1.85)^0.2 - 1 = 1.1309 - 1 = 0.1309 or 13.09%. This means the investment grew at an average annual rate of 13.09% over the 5-year period. CAGR is useful for comparing investment performance over time as it smooths out volatility and provides a single growth rate that represents the investment's performance.",
            "category": "financial_analysis"
        }
    ]


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process fintech training data")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input data file (JSON or JSONL)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['alpaca', 'chatml', 'simple'],
        default='alpaca',
        help="Instruction format"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample fintech data"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for evaluation"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FintechDataProcessor(instruction_format=args.format)
    
    if args.create_sample:
        logger.info("Creating sample fintech data...")
        sample_data = create_sample_fintech_data()
        
        # Save raw sample data
        os.makedirs("data/raw", exist_ok=True)
        sample_path = "data/raw/sample_fintech_data.json"
        with open(sample_path, 'w') as f:
            json.dump({"examples": sample_data}, f, indent=2)
        
        logger.info(f"Sample data saved to {sample_path}")
        
        # Process sample data
        result = processor.process_and_save(
            input_path=sample_path,
            output_dir=args.output_dir,
            eval_ratio=args.eval_ratio
        )
    elif args.input:
        result = processor.process_and_save(
            input_path=args.input,
            output_dir=args.output_dir,
            eval_ratio=args.eval_ratio
        )
    else:
        parser.print_help()
        return
    
    print("\n" + "="*50)
    print("DATA PROCESSING COMPLETE")
    print("="*50)
    print(f"Training data: {result['train_path']} ({result['train_count']} examples)")
    print(f"Evaluation data: {result['eval_path']} ({result['eval_count']} examples)")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

# Made with Bob
