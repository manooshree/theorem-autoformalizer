# import torch
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM,
#     Trainer, 
#     TrainingArguments, 
#     TrainerCallback
# )
# from datasets import load_dataset
# import os
# import sys
# import numpy as np
# import gc

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Dataset config
# dataset = "proofnet"
# # Print system info
# print("\nSystem Information:")
# print(f"Python version: {sys.version}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA device: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# # Load dataset
# print("\nLoading dataset...")
# data = load_dataset("json", data_files="/home/manooshree/MyITP/models/proofdef/proofnet.jsonl")

# print("\nDataset Statistics:")
# print(f"Total examples: {len(data['train'])}")
# print("Sample lengths:")

# if dataset == "forml4":
#     sample_lengths_nl = [len(ex.split()) for ex in data['train'][:5]['nl_problem']]
#     sample_lengths_formal = [len(ex.split()) for ex in data['train'][:5]['formal']]
# if dataset == "proofnet":
#     sample_lengths_nl = [len(ex.split()) for ex in data['train'][:5]['informal_prefix']]
#     sample_lengths_formal = [len(ex.split()) for ex in data['train'][:5]['formal_statement']]


# print(f"NL problems (first 5): {sample_lengths_nl}")
# print(f"Formal problems (first 5): {sample_lengths_formal}")

# # Model configuration
# model_name = "EleutherAI/llemma_7b"

# # Load tokenizer and model
# print("\nLoading tokenizer and model...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     trust_remote_code=True, 
#     use_cache=False
# )

# # Add padding token if it doesn't exist
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id

# print("Model loaded!")

# def preprocess_function(examples):
#     if dataset == "forml4":
#         inputs = [f"Translate NL to FL in Lean:\nInput: {ex}\nOutput: " for ex in examples["nl_problem"]]
#         outputs = [f"{ex}{tokenizer.eos_token}" for ex in examples["formal"]]

#     if dataset == "proofnet":
#         inputs = [f"Translate NL to FL in Lean:\nInput: {ex}\nOutput: " for ex in examples["informal_prefix"]]
#         outputs = [f"{ex}{tokenizer.eos_token}" for ex in examples["formal_statement"]]


    
#     # Combine input and output for causal language modeling
#     combined_texts = [inp + out for inp, out in zip(inputs, outputs)]
    
#     model_inputs = tokenizer(
#         combined_texts,
#         max_length=512,
#         truncation=True,
#         padding="max_length",
#         return_tensors="pt"
#     )
    
#     # Create attention mask and labels
#     labels = model_inputs["input_ids"].clone()
    
#     # Find the position where the output starts for each example
#     for idx, (inp, out) in enumerate(zip(inputs, outputs)):
#         input_len = len(tokenizer(inp)["input_ids"])
#         # Mask out the input portion in labels
#         labels[idx, :input_len] = -100
    
#     model_inputs["labels"] = labels
#     return model_inputs

# print("\nProcessing dataset...")
# tokenized_data = data['train'].map(preprocess_function, batched=True, remove_columns=data['train'].column_names)
# print("Dataset tokenized!")

# # Print tokenized data statistics
# print("\nTokenized Data Statistics:")
# print(f"Training examples: {len(tokenized_data)}")

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./llemma_nl_fl_theorem",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     num_train_epochs=20,
#     learning_rate=1e-5,
#     weight_decay=0.005,
#     logging_dir='./logs',
#     logging_steps=1,
#     save_steps=20,
#     save_total_limit=3,
#     fp16=True,
#     max_grad_norm=0.5,
#     lr_scheduler_type="cosine",
#     save_strategy="steps",
#     gradient_checkpointing=True,
#     report_to="tensorboard", 
#     gradient_checkpointing_kwargs={"use_reentrant": False}, 
#     optim="adamw_torch_fused" 
# )

# class GradientMonitorCallback(TrainerCallback):
#     def __init__(self):
#         self.training_loss_history = []
#         self.step_losses = []
        
#     def on_step_end(self, args, state, control, model=None, logs=None, **kwargs):
#         if logs is not None and "loss" in logs:
#             self.step_losses.append(logs["loss"])
        
#         if state.global_step % 1 == 0:
#             total_norm = 0
#             for p in model.parameters():
#                 if p.grad is not None:
#                     param_norm = p.grad.data.norm(2)
#                     total_norm += param_norm.item() ** 2
#             total_norm = total_norm ** (1. / 2)
            
#             # Calculate recent average loss
#             recent_avg_loss = sum(self.step_losses[-5:]) / len(self.step_losses[-5:]) if self.step_losses else 0
            
#             print(f'Step {state.global_step}:')
#             print(f'  Gradient norm: {total_norm:.4f}')
#             print(f'  Recent avg loss: {recent_avg_loss:.4f}')
            
#             # Monitor for training issues
#             if total_norm < 1e-4:
#                 print("  ⚠️ Warning: Possible vanishing gradient!")
#             elif total_norm > 5:
#                 print("  ⚠️ Warning: Possible exploding gradient!")
            
#             if torch.isnan(torch.tensor(total_norm)):
#                 print("  ⚠️ Critical: NaN gradient detected!")
                
#             # Monitor loss stability
#             if len(self.step_losses) > 10:
#                 loss_std = np.std(self.step_losses[-10:])
#                 if loss_std > 2.0:
#                     print("  ⚠️ Warning: Unstable loss detected!")

# print("\nInitializing trainer...")

# gc.collect()
# torch.cuda.empty_cache()

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_data,
#     callbacks=[GradientMonitorCallback()]
# )

# print("\nStarting training...")
# try:
#     trainer.train()
#     print("\nTraining complete!")
    
#     # Save the final model
#     print("\nSaving model...")
#     trainer.save_model("./final_llemma_model")
#     print("Model saved!")
    
# except Exception as e:
#     print(f"\nError during training: {str(e)}")
    
# finally:
#     # Print final training statistics
#     if hasattr(trainer, "state"):
#         print("\nFinal Training Statistics:")
#         print(f"Total steps: {trainer.state.global_step}")
#         if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
#             final_loss = trainer.state.log_history[-1].get('loss', 'N/A')
#             print(f"Final training loss: {final_loss}")


import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments, 
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset
import os
import sys
import numpy as np

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print system info
print("\nSystem Information:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Load and validate dataset
print("\nLoading dataset...")
data = load_dataset("json", data_files="/home/manooshree/MyITP/models/proofdef/forml4_regex.json")

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Total examples: {len(data['train'])}")
print("Sample lengths:")
sample_lengths_nl = [len(ex.split()) for ex in data['train'][:5]['nl_problem']]
sample_lengths_formal = [len(ex.split()) for ex in data['train'][:5]['formal']]
print(f"NL problems (first 5): {sample_lengths_nl}")
print(f"Formal problems (first 5): {sample_lengths_formal}")

# Model configuration
model_name = "EleutherAI/llemma_7b"

# Load tokenizer and model
print("\nLoading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# Configure LoRA with rank 128
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,  # higher rank for more capacity
    lora_alpha=256,  # 2x the rank
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # target attention layers
    modules_to_save=None  # don't save any modules fully
)

# Prepare model for training
print("\nPreparing model for LoRA fine-tuning...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Model loaded!")

def preprocess_function(examples):
    # Format prompt template
    inputs = [f"Translate NL to FL in Lean:\nInput: {ex}\nOutput: " for ex in examples["nl_problem"]]
    outputs = [f"{ex}{tokenizer.eos_token}" for ex in examples["formal"]]
    
    # Combine input and output for causal language modeling
    combined_texts = [inp + out for inp, out in zip(inputs, outputs)]
    
    model_inputs = tokenizer(
        combined_texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create attention mask and labels
    labels = model_inputs["input_ids"].clone()
    
    # Find the position where the output starts for each example
    for idx, (inp, out) in enumerate(zip(inputs, outputs)):
        input_len = len(tokenizer(inp)["input_ids"])
        # Mask out the input portion in labels
        labels[idx, :input_len] = -100
    
    model_inputs["labels"] = labels
    return model_inputs

print("\nProcessing dataset...")
tokenized_data = data['train'].map(preprocess_function, batched=True, remove_columns=data['train'].column_names)
print("Dataset tokenized!")

# Print tokenized data statistics
print("\nTokenized Data Statistics:")
print(f"Training examples: {len(tokenized_data)}")

# Training arguments optimized for rank 128 LoRA
training_args = TrainingArguments(
    output_dir="./llemma_lora_nl_fl_theorem",
    per_device_train_batch_size=2,  # reduced for higher rank
    gradient_accumulation_steps=4,  # increased for stability
    num_train_epochs=20,
    learning_rate=1e-4,  # slightly lower for higher rank
    weight_decay=0.01,  # increased for better regularization
    logging_dir='./logs',
    logging_steps=1,
    save_steps=20,
    save_total_limit=3,
    fp16=True,
    max_grad_norm=1.0,  # increased for higher rank
    warmup_ratio=0.05,  # added warmup
    lr_scheduler_type="cosine",
    save_strategy="steps",
    report_to="tensorboard",
    optim="adamw_torch_fused",
    gradient_checkpointing=True  # added for memory efficiency
)

class GradientMonitorCallback(TrainerCallback):
    def __init__(self):
        self.training_loss_history = []
        self.step_losses = []
        
    def on_step_end(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.step_losses.append(logs["loss"])
        
        if state.global_step % 1 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Calculate recent average loss
            recent_avg_loss = sum(self.step_losses[-5:]) / len(self.step_losses[-5:]) if self.step_losses else 0
            
            print(f'Step {state.global_step}:')
            print(f'  Gradient norm: {total_norm:.4f}')
            print(f'  Recent avg loss: {recent_avg_loss:.4f}')

print("\nInitializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    callbacks=[GradientMonitorCallback()]
)

print("\nStarting training...")
try:
    trainer.train()
    print("\nTraining complete!")
    
    # Save the final model
    print("\nSaving model...")
    model.save_pretrained("./final_llemma_lora_model")
    print("Model saved!")
    
except Exception as e:
    print(f"\nError during training: {str(e)}")
    
finally:
    # Print final training statistics
    if hasattr(trainer, "state"):
        print("\nFinal Training Statistics:")
        print(f"Total steps: {trainer.state.global_step}")
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            final_loss = trainer.state.log_history[-1].get('loss', 'N/A')
            print(f"Final training loss: {final_loss}")