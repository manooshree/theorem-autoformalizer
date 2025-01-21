import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    ByT5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()






def get_dir_size(dir_path):
    try:
        jsonl_files = glob(os.path.join(dir_path, "**/*.jsonl"), recursive=True)
        return sum(os.path.getsize(f) for f in jsonl_files) / (1024 * 1024)  # Size in MB
    except Exception as e:
        return 0


@dataclass
class PretrainingConfig:
    """Configuration for ByT5 pretraining"""
    # Model configuration
    model_name: str = "google/byt5-small"
    max_length: int = 768  # This is fine given your GPU memory
    
    # Memory settings for 46GB L40S
    batch_size: int = 4  # We can use a larger batch size
    gradient_accumulation_steps: int = 16  # Keep this moderate
    
    # Other training configuration
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    train_data_dir: str = "/home/manooshree/lean-work/ntp-toolkit/Examples/Mathlib/FullProof"  # Added this line
    eval_data_dir: Optional[str] = None 

    print(get_dir_size(train_data_dir))
   
    # Hardware configuration
    device: str = device
    num_workers: int = 4
    
    # Memory optimization flags
    fp16: bool = True  # Keep mixed precision training
    gradient_checkpointing: bool = True  # We can disable this given the GPU memory
    
    # Logging & Checkpointing
    output_dir: str = "byt5_lean_pretrain_output"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000

class LeanDataset(Dataset):
    """Custom dataset for ByT5 pretraining on Lean code"""
    def __init__(self, data_dir: str, tokenizer: ByT5Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Debug: Print the directory we're looking in
        logging.info(f"Initializing dataset from directory: {data_dir}")
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            logging.error(f"Directory does not exist: {data_dir}")
            return
        
        # Find all JSONL files recursively
        jsonl_files = glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
        logging.info(f"Found {len(jsonl_files)} JSONL files")
        
        # Print first few files found (if any)
        if jsonl_files:
            logging.info("First 3 files found:")
            for f in jsonl_files[:3]:
                logging.info(f"  - {f}")
        
        # Load JSONL data from all files
        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read line by line for JSONL
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            self.examples.append(data)
                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing line in {file_path}: {e}")
                            continue
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                continue
                
        logging.info(f"Total examples loaded: {len(self.examples)}")
        if len(self.examples) == 0:
            logging.error("No examples were loaded! Dataset will be empty.")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Combine all available fields into structured text
        text = (
        f"# File Information\n"
        f"Module: {example['module']}\n"
        f"File: {example['file']}\n\n"
        f"# Source Context\n"
        f"{example['srcUpToDecl']}\n\n"
        f"# Theorem\n"
        f"Name: {example['declName']}\n"
        f"{example['decl']}\n\n"
        f"# Proof\n"
        f"{example['proof']}"
    )
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = encoding['input_ids'].clone()
    
    # NEW: Return specific keys needed for seq2seq
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
def setup_pretraining(config: PretrainingConfig):
    """Setup all components for pretraining"""
    
    # Load tokenizer and model
    tokenizer = ByT5Tokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name) 

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    model = model.to(config.device)
    
    # Create datasets
    train_dataset = LeanDataset(
        config.train_data_dir,
        tokenizer,
        config.max_length
    )
    
    eval_dataset = None
    if config.eval_data_dir:
        eval_dataset = LeanDataset(
            config.eval_data_dir,
            tokenizer,
            config.max_length
        )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        evaluation_strategy="no",
        save_total_limit=3,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        # Memory optimizations
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )
    
    # Create data collator with span masking for pretraining
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model = model, 
        padding = True
    )
    
    return model, tokenizer, train_dataset, eval_dataset, training_args, data_collator

def main():
    # Setup logging
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    # Log system info
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    
    # Create config
    config = PretrainingConfig(
        max_length=768,
        batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_epochs=10,  # You might want to increase this since you have limited data
        warmup_steps=1000,
        fp16=True,  # Change to False if you want fp32
        train_data_dir="/home/manooshree/lean-work/ntp-toolkit/Examples/Mathlib/FullProof", 
        output_dir="byt5_lean_pretrain", 
        gradient_checkpointing = True
    )
    
    # Setup components
    model, tokenizer, train_dataset, eval_dataset, training_args, data_collator = setup_pretraining(config)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    logging.info(f"Training completed. Model saved to {final_output_dir}")

if __name__ == "__main__":
    main()