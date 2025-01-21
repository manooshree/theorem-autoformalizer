import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_data = load_dataset("json", data_files="test.jsonl")

# Load model and tokenizer
model_path = "./final_model"
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)

def preprocess_function(examples):
    inputs = ["Translate NL to FL in Lean: " + ex for ex in examples["informal_prefix"]]
    outputs = [ex for ex in examples["formal_statement"]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=150,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    labels = tokenizer(
        outputs,
        max_length=150,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    labels_with_ignore_index = []
    for label_seq in labels["input_ids"]:
        labels_with_ignore_index.append([
            label if label != tokenizer.pad_token_id else -100 
            for label in label_seq
        ])
    
    model_inputs["labels"] = torch.tensor(labels_with_ignore_index, dtype=torch.long)
    return model_inputs

print(f"\nLoaded test dataset with {len(test_data['train'])} examples")

def get_predictions(model, dataset):
    model.eval()
    predictions = []
    print(f"\nGenerating predictions for {len(dataset['train'])} examples...")
    
    with torch.no_grad():
        for idx in range(len(dataset['train'])):
            print(f"\nProcessing example {idx+1}/{len(dataset['train'])}...")
            
            # Debug prints
            print("\nDEBUG:")
            print("Original input:", dataset['train'][idx]['informal_prefix'][:100], "...")
            
            input_text = "Translate NL to FL in Lean: " + dataset['train'][idx]['informal_prefix']
            print("Model input:", input_text[:100], "...")
            
            model_inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=150,
                truncation=True,
                padding="max_length"
            ).to(device)
            
            # Print tokenized input
            print("Tokenized input:", tokenizer.decode(model_inputs['input_ids'][0]), "...")
            
            generated = model.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                max_length=150,
                num_beams=5,
                early_stopping=True,
                do_sample=False,
                temperature=0.0
            )
            
            pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            predictions.append(pred_text)
            
            # Print prediction
            print("Prediction:", pred_text[:100], "...")
            
    print(f"\nGenerated {len(predictions)} predictions")
    return predictions

# Preprocess test dataset for loss calculation
tokenized_test = test_data.map(preprocess_function, batched=True)

# Get predictions
predictions = get_predictions(model, test_data)

# Set up evaluation trainer for loss calculation
eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=2,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_test["train"],
)

# Get overall loss
eval_results = trainer.evaluate()
print(f"\nNumber of predictions made: {len(predictions)}")

# Print results
print("\nPer-Example Evaluation Results:")
print("-" * 50)
for idx, pred in enumerate(predictions):
    print(f"\nExample {idx + 1}:")
    print(f"Input:     {test_data['train'][idx]['informal_prefix']}")
    print(f"Target:    {test_data['train'][idx]['formal_statement']}")
    print(f"Predicted: {pred}")

print("\n" + "=" * 50)
print(f"Test Loss: {eval_results['eval_loss']:.3f}")

# Save results to file
with open("byt5_evaluation_results.txt", "w") as f:
    f.write("Per-Example Evaluation Results:\n")
    f.write("-" * 50 + "\n")
    
    for idx, pred in enumerate(predictions):
        f.write(f"\nExample {idx + 1}:\n")
        f.write(f"Input:     {test_data['train'][idx]['informal_prefix']}\n")
        f.write(f"Target:    {test_data['train'][idx]['formal_statement']}\n")
        f.write(f"Predicted: {pred}\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Test Loss: {eval_results['eval_loss']:.3f}\n")

print("\nDetailed results saved to byt5_evaluation_results.txt")