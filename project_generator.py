"""
Project Generator for Smart AI Model Builder
============================================

This module handles the generation of complete AI projects with all necessary files.
It creates a structured folder with code, dependencies, and documentation.
It also includes a detailed logging system for tracking project progress.
- Make By Mohamed Sarhan
=======================================================
"""

import os
import json
import logging
from datetime import datetime
import shutil

class ProjectGenerator:
    """Project generator for creating complete AI model projects"""
    
    def __init__(self, gui_logger=None):
        """Initialize the project generator"""
        self.logger = logging.getLogger(__name__)
        self.gui_logger = gui_logger
        self.project_root = "TINN"  # The Intelligent Neural Network
        
    def log_message(self, message, level="INFO"):
        """Log a message to both console and GUI"""
        self.logger.info(message)
        if self.gui_logger:
            self.gui_logger(message, level)
            
    def create_project_structure(self, project_name):
        """
        Create the project folder structure
        
        Args:
            project_name (str): Name of the project
            
        Returns:
            str: Path to the project root directory
        """
        self.log_message(f"Creating project structure for: {project_name}")
        
        # Create main project directory
        project_path = os.path.join(self.project_root, project_name)
        os.makedirs(project_path, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "src", 
            "data", 
            "models", 
            "configs", 
            "docs", 
            "tests", 
            "notebooks", 
            "scripts",
            "utils",
            "assets",
            "logs"
        ]
        for subdir in subdirs:
            os.makedirs(os.path.join(project_path, subdir), exist_ok=True)
            
        self.log_message(f"Project structure created at: {project_path}")
        return project_path
        
    def generate_main_script(self, project_path, model_config):
        """
        Generate the main application script
        
        Args:
            project_path (str): Path to the project directory
            model_config (dict): Model configuration information
        """
        self.log_message("Generating main application script")
        
        main_script_content = f'''#!/usr/bin/env python3
"""
Main Application Script for {model_config['custom_name']}
=====================================================

This is an automatically generated AI application based on user requirements.
Model: {model_config['model_id']}
Author: {model_config['author']}
Description: {model_config['description']}
"""

import os
import sys
import logging
import json
from datetime import datetime
import argparse

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required libraries
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    import torch
    import numpy as np
except ImportError as e:
    print(f"Required library not found: {{e}}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

class {model_config['custom_name'].replace(' ', '').replace('-', '')}AI:
    """Main AI application class"""
    
    def __init__(self, model_path=None):
        """Initialize the AI application"""
        self.model_id = "{model_config['model_id']}"
        self.model = None
        self.tokenizer = None
        self.setup_logging()
        
        if model_path and os.path.exists(model_path):
            self.load_custom_model(model_path)
        else:
            self.load_model()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/app.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing {{self.__class__.__name__}}")
        
    def load_model(self):
        """Load the AI model from Hugging Face"""
        try:
            self.logger.info(f"Loading model: {{self.model_id}}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Try to load as a sequence classification model first
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            except:
                # Fall back to base model
                self.model = AutoModel.from_pretrained(self.model_id)
                
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {{e}}")
            raise
            
    def load_custom_model(self, model_path):
        """Load a custom trained model"""
        try:
            self.logger.info(f"Loading custom model from: {{model_path}}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.logger.info("Custom model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading custom model: {{e}}")
            self.logger.info("Falling back to base model")
            self.load_model()
            
    def process_input(self, input_text):
        """
        Process input text with the AI model
        
        Args:
            input_text (str): Input text to process
            
        Returns:
            dict: Processing results
        """
        try:
            self.logger.info("Processing input text")
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            
            # Process with model
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Return results
            results = {{
                "input": input_text,
                "model_id": self.model_id,
                "output_shape": str(outputs.last_hidden_state.shape) if hasattr(outputs, 'last_hidden_state') else str(outputs.logits.shape),
                "processed_at": datetime.now().isoformat()
            }}
            
            self.logger.info("Input processed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing input: {{e}}")
            return {{"error": str(e)}}
            
    def train_model(self, training_data_path, epochs=3):
        """
        Train the model with custom data
        
        Args:
            training_data_path (str): Path to training data
            epochs (int): Number of training epochs
        """
        self.logger.info(f"Starting model training with data from: {{training_data_path}}")
        
        try:
            # This is a simplified training example
            # In a real implementation, you would load and process your dataset
            self.logger.info(f"Training for {{epochs}} epochs...")
            
            # Simulate training time
            import time
            for epoch in range(epochs):
                self.logger.info(f"Epoch {{epoch+1}}/{{epochs}}")
                time.sleep(2)  # Simulate training time
                
            self.logger.info("Model training completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during training: {{e}}")
            return False
            
    def save_model(self, save_path):
        """
        Save the trained model
        
        Args:
            save_path (str): Path to save the model
        """
        try:
            self.logger.info(f"Saving model to: {{save_path}}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {{e}}")
            
    def run_interactive(self):
        """Run the main application in interactive mode"""
        self.logger.info("Starting interactive mode")
        print("AI Application is ready!")
        print("Enter 'quit' to exit.")
        
        while True:
            try:
                user_input = input("\\n> ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                result = self.process_input(user_input)
                print(json.dumps(result, indent=2))
                
            except KeyboardInterrupt:
                print("\\nExiting application...")
                break
            except Exception as e:
                print(f"Error: {{e}}")
                
        self.logger.info("Interactive mode terminated")
        
    def run_batch(self, input_file, output_file):
        """
        Process inputs from a file in batch mode
        
        Args:
            input_file (str): Path to input file
            output_file (str): Path to output file
        """
        self.logger.info(f"Starting batch processing: {{input_file}} -> {{output_file}}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                inputs = [line.strip() for line in f if line.strip()]
                
            results = []
            for i, input_text in enumerate(inputs):
                self.logger.info(f"Processing item {{i+1}}/{{len(inputs)}}")
                result = self.process_input(input_text)
                results.append({{
                    "input": input_text,
                    "output": result
                }})
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Batch processing completed. Results saved to {{output_file}}")
            
        except Exception as e:
            self.logger.error(f"Error during batch processing: {{e}}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='{model_config['custom_name']} AI Application')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                        help='Run mode: interactive or batch')
    parser.add_argument('--input', help='Input file for batch mode')
    parser.add_argument('--output', help='Output file for batch mode')
    parser.add_argument('--model-path', help='Path to custom trained model')
    
    args = parser.parse_args()
    
    app = {model_config['custom_name'].replace(' ', '').replace('-', '')}AI(model_path=args.model_path)
    
    if args.mode == 'batch':
        if not args.input or not args.output:
            print("Batch mode requires --input and --output arguments")
            sys.exit(1)
        app.run_batch(args.input, args.output)
    else:
        app.run_interactive()

if __name__ == "__main__":
    main()
'''

        # Write to file
        main_script_path = os.path.join(project_path, "src", "main.py")
        with open(main_script_path, "w", encoding="utf-8") as f:
            f.write(main_script_content)
            
        self.log_message(f"Main script generated at: {main_script_path}")
        
    def generate_training_script(self, project_path, model_config):
        """
        Generate a training script for custom model training
        
        Args:
            project_path (str): Path to the project directory
            model_config (dict): Model configuration information
        """
        self.log_message("Generating training script")
        
        training_script_content = f'''#!/usr/bin/env python3
"""
Training Script for {model_config['custom_name']}
=========================================

This script trains a custom model based on user requirements.
"""

import os
import sys
import logging
import json
import argparse
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required libraries
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        Trainer, 
        TrainingArguments,
        DataCollatorWithPadding
    )
    from datasets import Dataset
    import torch
except ImportError as e:
    print(f"Required library not found: {{e}}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

class ModelTrainer:
    """Custom model trainer class"""
    
    def __init__(self, model_id, output_dir="./models"):
        """Initialize the trainer"""
        self.model_id = model_id
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ModelTrainer")
        
    def load_base_model(self):
        """Load the base model and tokenizer"""
        try:
            self.logger.info(f"Loading base model: {{self.model_id}}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.logger.info("Base model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading base model: {{e}}")
            raise
            
    def prepare_dataset(self, data_path):
        """
        Prepare dataset for training
        
        Args:
            data_path (str): Path to training data
            
        Returns:
            Dataset: Prepared dataset
        """
        self.logger.info(f"Preparing dataset from: {{data_path}}")
        
        try:
            # Load data (assuming JSON format)
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert to Hugging Face Dataset format
            dataset = Dataset.from_dict({{
                "text": [item["input"] for item in data],
                "labels": [item["output"] for item in data]
            }})
            
            # Tokenize the dataset
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], truncation=True, padding=True)
                
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            self.logger.info(f"Dataset prepared with {{len(dataset)}} samples")
            return tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {{e}}")
            raise
            
    def train(self, train_dataset, eval_dataset=None, epochs=3, batch_size=8):
        """
        Train the model
        
        Args:
            train_dataset (Dataset): Training dataset
            eval_dataset (Dataset): Evaluation dataset (optional)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        self.logger.info("Starting model training")
        
        try:
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(self.output_dir, 'logs'),
                logging_steps=10,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if eval_dataset else False,
            )
            
            # Define data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            # Start training
            trainer.train()
            
            # Save the trained model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during training: {{e}}")
            raise
            
    def evaluate(self, test_dataset):
        """
        Evaluate the trained model
        
        Args:
            test_dataset (Dataset): Test dataset
            
        Returns:
            dict: Evaluation results
        """
        self.logger.info("Evaluating model")
        
        try:
            trainer = Trainer(model=self.model, tokenizer=self.tokenizer)
            results = trainer.evaluate(test_dataset)
            self.logger.info(f"Evaluation results: {{results}}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {{e}}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train custom {model_config['custom_name']} model')
    parser.add_argument('--model-id', default='{model_config['model_id']}', help='Base model ID')
    parser.add_argument('--data-path', required=True, help='Path to training data (JSON format)')
    parser.add_argument('--output-dir', default='./models/trained_model', help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(args.model_id, args.output_dir)
    
    # Load base model
    trainer.load_base_model()
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(args.data_path)
    
    # Split dataset (80% train, 20% eval)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"Model training completed. Trained model saved to {{args.output_dir}}")

if __name__ == "__main__":
    main()
'''

        # Write to file
        training_script_path = os.path.join(project_path, "src", "train.py")
        with open(training_script_path, "w", encoding="utf-8") as f:
            f.write(training_script_content)
            
        self.log_message(f"Training script generated at: {training_script_path}")
        
    def generate_requirements(self, project_path, model_config):
        """
        Generate the requirements.txt file
        
        Args:
            project_path (str): Path to the project directory
            model_config (dict): Model configuration information
        """
        self.log_message("Generating requirements file")
        
        # Base requirements
        requirements = [
            "torch>=1.9.0",
            "transformers>=4.10.0",
            "tokenizers>=0.10.0",
            "numpy>=1.21.0",
            "requests>=2.25.0",
            "datasets>=2.0.0",
            "accelerate>=0.12.0"
        ]
        
        # Add library-specific requirements
        library_name = model_config.get("library_name", "").lower()
        if "tensorflow" in library_name:
            requirements.append("tensorflow>=2.6.0")
        elif "flax" in library_name:
            requirements.append("jax>=0.2.0")
            requirements.append("flax>=0.3.0")
            
        # Add additional APIs
        for api in model_config.get("additional_apis", []):
            if "openai" in api.lower():
                requirements.append("openai>=0.27.0")
            elif "stability" in api.lower():
                requirements.append("stability-sdk>=0.2.0")
                
        # Write to file
        requirements_path = os.path.join(project_path, "requirements.txt")
        with open(requirements_path, "w", encoding="utf-8") as f:
            f.write("\\n".join(requirements))
            
        self.log_message(f"Requirements file generated at: {requirements_path}")
        
    def generate_readme(self, project_path, model_config):
        """
        Generate the README.md file
        
        Args:
            project_path (str): Path to the project directory
            model_config (dict): Model configuration information
        """
        self.log_message("Generating README file")
        
        readme_content = f'''# {model_config['custom_name']}

## Description
{model_config['description']}

This is an automatically generated AI application based on user requirements using the Smart AI Model Builder.

## Model Information
- **Model ID**: {model_config['model_id']}
- **Author**: {model_config['author']}
- **Features**: {', '.join(model_config.get('features', []))}

## Setup Instructions

### Automatic Installation (Windows)
Run the installation script:
```bash
install.bat
```

### Manual Installation
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application in interactive mode:
   ```bash
   python src/main.py
   ```

3. Run the application in batch mode:
   ```bash
   python src/main.py --mode batch --input data/input.txt --output data/output.json
   ```

4. Train a custom model:
   ```bash
   python src/train.py --data-path data/training_data.json --epochs 5
   ```

## Usage

### Interactive Mode
After running the application in interactive mode, you can enter text prompts and the AI will process them using the {model_config['model_id']} model.

Type 'quit' to exit the application.

### Batch Mode
Create a text file with one input per line in `data/input.txt`, then run:
```bash
python src/main.py --mode batch --input data/input.txt --output data/output.json
```

### Training Custom Models
Prepare your training data in JSON format and run:
```bash
python src/train.py --data-path data/training_data.json --epochs 5
```

## Project Structure

```
{model_config['custom_name']}/
├── src/
│   ├── main.py          # Main application script
│   └── train.py         # Model training script
├── data/                # Data files
│   ├── input.txt        # Sample input file for batch processing
│   ├── output.json      # Sample output file from batch processing
│   └── training_data.json # Sample training data
├── models/              # Model files
│   └── trained_model/   # Custom trained models
├── configs/             # Configuration files
├── docs/                # Documentation
├── tests/               # Test files
├── notebooks/           # Jupyter notebooks
├── scripts/             # Utility scripts
├── utils/               # Utility functions
├── assets/              # Asset files
├── logs/                # Log files
├── requirements.txt     # Python dependencies
├── install.bat          # Windows installation script
├── install.sh           # Linux/Mac installation script
├── run.bat              # Windows run script
├── run.sh               # Linux/Mac run script
└── README.md            # This file
```

## API Reference

### Main Application (`src/main.py`)
- `--mode`: Run mode (interactive or batch)
- `--input`: Input file for batch mode
- `--output`: Output file for batch mode
- `--model-path`: Path to custom trained model

### Training Script (`src/train.py`)
- `--model-id`: Base model ID (default: {model_config['model_id']})
- `--data-path`: Path to training data (JSON format)
- `--output-dir`: Output directory for trained model
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size

## Generated On
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Additional APIs
{', '.join(model_config.get('additional_apis', ['None']))}

## License
This project is licensed under the MIT License.
'''

        # Write to file
        readme_path = os.path.join(project_path, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
            
        self.log_message(f"README file generated at: {readme_path}")
        
    def generate_sample_data(self, project_path, model_config):
        """
        Generate sample data files
        
        Args:
            project_path (str): Path to the project directory
            model_config (dict): Model configuration information
        """
        self.log_message("Generating sample data files")
        
        # Create sample input file
        sample_input_path = os.path.join(project_path, "data", "input.txt")
        with open(sample_input_path, "w", encoding="utf-8") as f:
            f.write("What is artificial intelligence?\\n")
            f.write("How does machine learning work?\\n")
            f.write("Explain neural networks\\n")
            f.write("What are the applications of AI?\\n")
            
        # Create sample training data
        sample_training_data = [
            {
                "input": "What is artificial intelligence?",
                "output": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines."
            },
            {
                "input": "How does machine learning work?",
                "output": "Machine learning is a subset of AI that enables computers to learn from data."
            }
        ]
        
        sample_training_path = os.path.join(project_path, "data", "training_data.json")
        with open(sample_training_path, "w", encoding="utf-8") as f:
            json.dump(sample_training_data, f, indent=2, ensure_ascii=False)
            
        self.log_message(f"Sample data files generated")
        
    def generate_config(self, project_path, model_config):
        """
        Generate configuration files
        
        Args:
            project_path (str): Path to the project directory
            model_config (dict): Model configuration information
        """
        self.log_message("Generating configuration files")
        
        # Save model configuration as JSON
        config_path = os.path.join(project_path, "configs", "model_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)
            
        # Create training configuration
        training_config = {
            "model_id": model_config["model_id"],
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "output_dir": "./models/trained_model"
        }
        
        training_config_path = os.path.join(project_path, "configs", "training_config.json")
        with open(training_config_path, "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2)
            
        self.log_message(f"Configuration files generated")
        
    def generate_utility_scripts(self, project_path, model_config):
        """
        Generate utility scripts
        
        Args:
            project_path (str): Path to the project directory
            model_config (dict): Model configuration information
        """
        self.log_message("Generating utility scripts")
        
        # Generate a data preparation script
        prep_script_content = '''#!/usr/bin/env python3
"""
Data Preparation Script
=======================

This script helps prepare data for training custom models.
"""

import json
import argparse
import os

def convert_csv_to_json(csv_file, json_file):
    """Convert CSV to JSON format"""
    import csv
    
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Converted {len(data)} records from {csv_file} to {json_file}")

def main():
    parser = argparse.ArgumentParser(description='Data preparation utilities')
    parser.add_argument('--csv-to-json', nargs=2, metavar=('CSV_FILE', 'JSON_FILE'),
                        help='Convert CSV file to JSON')
    
    args = parser.parse_args()
    
    if args.csv_to_json:
        convert_csv_to_json(args.csv_to_json[0], args.csv_to_json[1])

if __name__ == "__main__":
    main()
'''
        
        prep_script_path = os.path.join(project_path, "scripts", "prepare_data.py")
        with open(prep_script_path, "w", encoding="utf-8") as f:
            f.write(prep_script_content)
            
        self.log_message(f"Utility scripts generated")
        
    def generate_installation_scripts(self, project_path):
        """
        Generate installation scripts for different platforms
        
        Args:
            project_path (str): Path to the project directory
        """
        self.log_message("Generating installation scripts")
        
        project_name = os.path.basename(project_path)
        
        # Windows batch script
        win_script_content = f'''@echo off
echo Installing {project_name}...
echo =========================

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment
    pause
    exit /b %errorlevel%
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\\Scripts\\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b %errorlevel%
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies
    pause
    exit /b %errorlevel%
)

echo.
echo Installation completed successfully!
echo To run the application, execute: run.bat
echo.
pause
'''
        
        win_script_path = os.path.join(project_path, "install.bat")
        with open(win_script_path, "w", encoding="utf-8") as f:
            f.write(win_script_content)
            
        # Linux/Mac shell script
        unix_script_content = f'''#!/bin/bash

echo "Installing {project_name}..."
echo "========================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies"
    exit 1
fi

echo ""
echo "Installation completed successfully!"
echo "To run the application, execute: ./run.sh"
'''
        
        unix_script_path = os.path.join(project_path, "install.sh")
        with open(unix_script_path, "w", encoding="utf-8") as f:
            f.write(unix_script_content)
            
        # Make shell script executable
        os.chmod(unix_script_path, 0o755)
        
        # Windows run script
        win_run_content = f'''@echo off
echo Running {project_name}...
echo ====================

REM Activate virtual environment
call venv\\Scripts\\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    echo Please run install.bat first
    pause
    exit /b %errorlevel%
)

REM Run the application
python src/main.py

pause
'''
        
        win_run_path = os.path.join(project_path, "run.bat")
        with open(win_run_path, "w", encoding="utf-8") as f:
            f.write(win_run_content)
            
        # Linux/Mac run script
        unix_run_content = f'''#!/bin/bash

echo "Running {project_name}..."
echo "===================="

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    echo "Please run ./install.sh first"
    exit 1
fi

# Run the application
python src/main.py
'''
        
        unix_run_path = os.path.join(project_path, "run.sh")
        with open(unix_run_path, "w", encoding="utf-8") as f:
            f.write(unix_run_content)
            
        # Make shell script executable
        os.chmod(unix_run_path, 0o755)
            
        self.log_message(f"Installation scripts generated")
        
    def generate_project(self, model_config):
        """
        Generate a complete AI project
        
        Args:
            model_config (dict): Model configuration information
            
        Returns:
            str: Path to the generated project
        """
        self.log_message("Starting project generation")
        
        # Create project structure
        project_name = model_config.get("custom_name", "AI_Model").replace(" ", "_")
        project_path = self.create_project_structure(project_name)
        
        # Generate all files
        self.generate_main_script(project_path, model_config)
        self.generate_training_script(project_path, model_config)
        self.generate_requirements(project_path, model_config)
        self.generate_readme(project_path, model_config)
        self.generate_sample_data(project_path, model_config)
        self.generate_config(project_path, model_config)
        self.generate_utility_scripts(project_path, model_config)
        self.generate_installation_scripts(project_path)
        
        self.log_message(f"Project generation complete: {project_path}")
        return project_path