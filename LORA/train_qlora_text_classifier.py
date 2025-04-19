import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchmetrics import Accuracy
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    DefaultDataCollator, get_cosine_schedule_with_warmup

# Import QLoRA instead of LoRA
from qlora import QLoRAConfig, QLoRAModel, QuantizationType, get_paged_optimizer

import warnings 
warnings.filterwarnings("ignore")

##########################
### TRAINING ARGUMENTS ###
##########################
experiment_name = "QLoRA_TextClassifier"
wandb_run_name = "bert_qlora_classifier"
working_directory = "work_dir"
epochs = 3
batch_size = 32  # Smaller batch size for QLoRA due to memory reasons
learning_rate = 2e-5  # Lower learning rate for QLoRA
weight_decay = 0.001
warmup_steps = 100
max_grad_norm = 1.0
num_workers = 8
gradient_checkpointing = True  # Enable gradient checkpointing to reduce memory usage
log_wandb = False
max_seq_length = 128
hf_dataset = "imdb"
hf_model_name = "bert-base-uncased"  # Can be replaced with larger models for QLoRA benefits

#######################
### QLoRA ARGUMENTS ###
#######################
use_qlora = True
train_head_only = False
target_modules = ["query", "key", "value", "dense", "output.dense"]
exclude_modules = ["classifier"]  # Don't do QLoRA on untrained classifier
rank = 8
lora_alpha = 16  # Higher alpha value for stability
use_rslora = True
bias = "none"
lora_dropout = 0.1
# QLoRA specific parameters
quantization_type = QuantizationType.NF4  # Normal Float 4-bit quantization
double_quant = True  # Enable double quantization for extra memory savings
paged_adamw = True  # Use paged optimizer to offload optimizer states

########################
### Init Accelerator ###
########################
path_to_experiment = os.path.join(working_directory, experiment_name)
if not os.path.isdir(path_to_experiment):
    os.makedirs(path_to_experiment, exist_ok=True)

accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if log_wandb else None)
if log_wandb:
    accelerator.init_trackers(experiment_name, init_kwargs={"wandb": {"name": wandb_run_name}})

###########################
### Prepare DataLoaders ###
###########################
dataset = load_dataset(hf_dataset)
labels = dataset["train"].features["label"].names if "label" in dataset["train"].features else ["negative", "positive"]

tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)

def preprocess_function(examples):
    # Tokenize the texts
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)
    return result

# Apply preprocessing to dataset
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on dataset",
)

# Prepare data loaders
collate_fn = DefaultDataCollator()
trainloader = DataLoader(
    tokenized_dataset["train"], 
    batch_size=batch_size, 
    collate_fn=collate_fn, 
    shuffle=True, 
    num_workers=num_workers
)
testloader = DataLoader(
    tokenized_dataset["test"], 
    batch_size=batch_size, 
    collate_fn=collate_fn, 
    shuffle=False, 
    num_workers=num_workers
)

#########################
### Load QLoRA Model ###
#########################
# Load base model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    hf_model_name, 
    num_labels=len(labels), 
    ignore_mismatched_sizes=True
)

# Enable gradient checkpointing to save memory
if gradient_checkpointing:
    model.gradient_checkpointing_enable()

# Handle non-QLORA training modes
if not use_qlora and train_head_only:
    accelerator.print("Training Classifier Head Only")
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

# Apply QLoRA to the model
if use_qlora:
    accelerator.print("Converting to QLoRA")
    qlora_config = QLoRAConfig(
        # LoRA parameters
        rank=rank, 
        target_modules=target_modules, 
        exclude_modules=exclude_modules, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        bias=bias, 
        use_rslora=use_rslora,
        # QLoRA parameters
        quantization=quantization_type,
        double_quant=double_quant,
        # Memory optimization
        gradient_checkpointing=gradient_checkpointing,
        paged_adamw=paged_adamw
    )

    # Convert model to QLoRA
    model = QLoRAModel(model, qlora_config).to(accelerator.device)

# Print the model architecture
accelerator.print(model)

###############################
### Define Training Metrics ###
###############################
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task="multiclass", num_classes=len(labels)).to(accelerator.device)

########################
### Define Optimizer ###
########################
# Use either paged optimizer or regular optimizer based on config
if use_qlora and paged_adamw:
    # Use paged optimizer for QLoRA to save memory by offloading optimizer states
    optimizer = get_paged_optimizer(
        model,
        lr=learning_rate,
        weight_decay=weight_decay
    )
else:
    # Regular optimizer for non-QLoRA or when paged optimizer not requested
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate, weight_decay=weight_decay)

########################
### Define Scheduler ###
########################
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, 
    num_warmup_steps=warmup_steps * accelerator.num_processes, 
    num_training_steps=epochs * len(trainloader) * accelerator.num_processes
)

##########################
### Prepare Everything ###
##########################
model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, testloader, scheduler
)

#####################
### Training Loop ###
#####################
for epoch in range(epochs):
    
    accelerator.print(f"Training Epoch {epoch}")

    ### Storage for Metrics ###
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    ### Training Progress Bar ###
    progress_bar = tqdm(range(len(trainloader)), disable=not accelerator.is_local_main_process)

    # Training phase
    model.train()
    for batch in trainloader:
        # Move data to device using accelerator
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None)
        )
        logits = outputs.logits
        
        # Compute loss
        loss = loss_fn(logits, batch["labels"])
        
        # Compute accuracy
        predicted = logits.argmax(dim=1)
        accuracy = accuracy_fn(predicted, batch["labels"])
        
        # Backward pass with accelerator
        accelerator.backward(loss)
        
        # Gradient clipping
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update model
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Gather metrics across GPUs
        loss_gathered = accelerator.gather_for_metrics(loss)
        accuracy_gathered = accelerator.gather_for_metrics(accuracy)
        
        # Store metrics
        train_loss.append(torch.mean(loss_gathered).item())
        train_acc.append(torch.mean(accuracy_gathered).item())
        
        # Update progress bar
        progress_bar.update(1)
        
        # Update learning rate
        scheduler.step()

    # Evaluation phase
    model.eval()
    for batch in tqdm(testloader, disable=not accelerator.is_local_main_process):
        # Move data to device
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        
        # Forward pass without gradient computation
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None)
            )
            logits = outputs.logits
        
        # Compute loss
        loss = loss_fn(logits, batch["labels"])
        
        # Compute accuracy
        predicted = logits.argmax(dim=1)
        accuracy = accuracy_fn(predicted, batch["labels"])
        
        # Gather metrics across GPUs
        loss_gathered = accelerator.gather_for_metrics(loss)
        accuracy_gathered = accelerator.gather_for_metrics(accuracy)
        
        # Store metrics
        test_loss.append(torch.mean(loss_gathered).item())
        test_acc.append(torch.mean(accuracy_gathered).item())
    
    # Calculate epoch metrics
    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_test_acc = np.mean(test_acc)
    
    # Print metrics
    accelerator.print(f"Training Accuracy: {epoch_train_acc:.4f}, Training Loss: {epoch_train_loss:.4f}")
    accelerator.print(f"Testing Accuracy: {epoch_test_acc:.4f}, Testing Loss: {epoch_test_loss:.4f}")
    
    # Log metrics with Weights and Biases if enabled
    accelerator.log({
        "training_loss": epoch_train_loss,
        "testing_loss": epoch_test_loss, 
        "training_acc": epoch_train_acc, 
        "testing_acc": epoch_test_acc
    }, step=epoch)

### Save Final Model ###
accelerator.wait_for_everyone()

# Save model based on training configuration
if use_qlora:
    # For QLoRA, save both adapter and merged versions
    accelerator.unwrap_model(model).save_model(
        os.path.join(working_directory, experiment_name, "qlora_adapter_checkpoint.safetensors")
    )
    accelerator.unwrap_model(model).save_model(
        os.path.join(working_directory, experiment_name, "qlora_merged_checkpoint.safetensors"), 
        merge_weights=True
    )
elif not use_qlora and train_head_only:
    # For head-only training
    save_file(
        accelerator.unwrap_model(model).state_dict(), 
        os.path.join(working_directory, experiment_name, "headonly_checkpoint.safetensors")
    )
else:
    # For full model training
    save_file(
        accelerator.unwrap_model(model).state_dict(), 
        os.path.join(working_directory, experiment_name, "fulltrain_checkpoint.safetensors")
    )

# End training (required for W&B)
accelerator.end_training() 