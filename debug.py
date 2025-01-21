import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = load_dataset("json", data_files="/home/manooshree/MyITP/models/proofdef/forml4_regex.json")


model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  #FP32
)

#FP32
for param in model.parameters():
    param.data = param.data.float()

print("model found!")


def preprocess_function(examples):
    inputs = ["Translate NL to FL in Lean: " + ex for ex in examples["nl_problem"]]
    outputs = [ex for ex in examples["formal"]]
    
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
    
    model_inputs["labels"] = torch.tensor(labels_with_ignore_index, dtype=torch.long)  # Specified dtype
    return model_inputs

print("dataset processed")
tokenized_data = data.map(preprocess_function, batched=True)
print("dataset tokenized")
print(tokenized_data["train"][0])
training_args = TrainingArguments(
    output_dir="./byt5_nl_fl_theorem",
    per_device_train_batch_size=2,        
    num_train_epochs=20,                 
    learning_rate=1e-5,                  
    weight_decay=0.005,                   
    logging_dir='./logs',
    # logging_steps=1,                      
    # save_steps=10,                        
    evaluation_strategy="no",
    save_total_limit=2,
    fp16=False,                          
    max_grad_norm=0.5,                   
    # warmup_steps=10,                     
    # gradient_accumulation_steps=4,      
    lr_scheduler_type="cosine",          
    # dataloader_num_workers=0,            
    save_strategy="epoch",     


    # TRAINING PARAMS FOR N = 16 BELOW
    # logging_steps=1
    # save_steps=10
    # warmup_steps=10
    # gradient_accumulation_steps=4
    # dataloader_num_workers=0

    # TRAINING PARAMS FOR N = 14000 BELOW
    logging_steps=50,
    save_steps=500,
    warmup_steps=100,
    gradient_accumulation_steps=4,
    dataloader_num_workers=4,
)

class GradientMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(f'Step {state.global_step}, Gradient norm: {total_norm}')
        if torch.isnan(torch.tensor(total_norm)):
            print("NaN gradient detected!")

print("trainer initialized")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    callbacks=[GradientMonitorCallback()]
)


trainer.train()

print("training complete")
trainer.save_model("./final_model") 
print("model saved!")
