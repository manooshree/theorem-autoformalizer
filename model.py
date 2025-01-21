import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset from a JSON file
data = load_dataset("json", data_files="/home/manooshree/MyITP/models/proofdef/train.jsonl")

# Initialize tokenizer and model
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
print("model found!")

def preprocess_function(examples):
    inputs = ["Translate NL to FL in Lean: " + ex for ex in examples["informal_prefix"]]
    outputs = [ex for ex in examples["formal_statement"]]
    model_inputs = tokenizer(inputs, max_length=150, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=150, truncation=True, padding="max_length")
    

    # labels["input_ids"] = [
    #     [(label if label != tokenizer.pad_token_id else -100) for label in labels]
    #     for labels in labels["input_ids"]
    # ]
    
    
    # model_inputs["labels"] = labels["input_ids"]

    labels_with_ignore_index = []
    for label_seq in labels["input_ids"]:
        labels_with_ignore_index.append([
            label if label != tokenizer.pad_token_id else -100 
            for label in label_seq
        ])
    
    model_inputs["labels"] = torch.tensor(labels_with_ignore_index)
    return model_inputs

print("dataset processed")
tokenized_data = data.map(preprocess_function, batched=True)
print("dataset tokenized")
print(tokenized_data["train"][0])
training_args = TrainingArguments(
    output_dir="./byt5_nl_fl_theorem",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="no",  
    save_total_limit=2,
    fp16=torch.cuda.is_available(), 
)

print("trainer initialized")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

# Start fine-tuning
trainer.train()

print("training complete")



# REFACTOR 

# pip install torch transformers datasets

# import torch
# from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
# from datasets import load_dataset

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load dataset from a JSON file
# data = load_dataset("json", data_files="/home/manooshree/MyITP/models/proofdef/train.jsonl")

# # Initialize tokenizer and model
# model_name = "google/byt5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# # model.to(device)  # Move model to the specified device
# print("Model loaded and moved to device:", device)

# # Define preprocessing function
# def preprocess_function(examples):
#     inputs = ["Translate NL to FL: " + ex for ex in examples["informal_prefix"]]
#     outputs = [ex for ex in examples["formal_statement"]]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#     labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length")
    
#     # Replace padding token ID with -100 to ignore padding in loss calculation
#     labels["input_ids"] = [
#     [label if label != tokenizer.pad_token_id else -100 for label in label_seq]
#     for label_seq in labels["input_ids"]
# ]

    
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# # Process dataset
# print("Processing dataset...")
# tokenized_data = data.map(preprocess_function, batched=True)
# print("Dataset tokenized!")

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./byt5_nl_fl_theorem",
#     per_device_train_batch_size=4,
#     num_train_epochs=10,
#     learning_rate=3e-5,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     save_steps=50,
#     evaluation_strategy="no",  
#     save_total_limit=2,
#     fp16=torch.cuda.is_available()
# )

# # Initialize Trainer
# print("Initializing Trainer...")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_data["train"],
# )

# # Fine-tuning
# print("Starting fine-tuning...")
# trainer.train()
# print("Training complete.")