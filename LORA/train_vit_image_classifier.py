import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from torchmetrics import Accuracy
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForImageClassification, AutoImageProcessor, \
    DefaultDataCollator, get_cosine_schedule_with_warmup

from lora import LoraConfig, LoraModel

import warnings 
warnings.filterwarnings("ignore")

##########################
### TRAINING ARGUMENTS ###
##########################
experiment_name = "LoRAImageClassifier"
wandb_run_name = "vit_lora_classifier"
working_directory = "work_dir"
epochs = 3
batch_size = 64
learning_rate = 3e-5
weight_decay = 0.001
warmup_steps = 100
max_grad_norm = 1.0
num_workers = 32
gradient_checkpointing = False
log_wandb = False
hf_dataset = "food101"
hf_model_name = "google/vit-base-patch16-224"

######################
### LORA ARGUMENTS ###
######################
use_lora = True
train_head_only = False
target_modules = ["query", "key", "value", "dense"]
exclude_modules = ["classifier"] # Dont do LoRA on untrained classifier
rank = 8
lora_alpha = 8
use_rslora = True
bias = "none"
lora_dropout = 0.1

########################
### Init Accelerator ###
########################
path_to_experiment = os.path.join(working_directory, experiment_name)
if not os.path.isdir(path_to_experiment):
    os.mkdir(path_to_experiment)

accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if log_wandb else None)
if log_wandb:
    accelerator.init_trackers(experiment_name, init_kwargs={"wandb": {"name": wandb_run_name}})


###########################
### Prepare DataLoaders ###
###########################
dataset = load_dataset(hf_dataset)
labels = dataset["train"].features["label"].names

processor = AutoImageProcessor.from_pretrained(hf_model_name, use_fast=True)

def transforms(examples):
    examples["pixel_values"] = [processor(img.convert("RGB"))["pixel_values"][0] for img in examples["image"]]
    del examples["image"]
    return examples

dataset = dataset.with_transform(transforms)

collate_fn = DefaultDataCollator()
trainloader = DataLoader(dataset["train"], batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
testloader = DataLoader(dataset["validation"], batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers)

#######################
### Load LoRA Model ###
#######################

model = AutoModelForImageClassification.from_pretrained(hf_model_name, 
                                                        num_labels=len(labels), 
                                                        ignore_mismatched_sizes=True)
if gradient_checkpointing:
    model.gradient_checkpointing_enable()

if not use_lora and train_head_only:
    accelerator.print("Training Classifier Head Only")
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

if use_lora:
    accelerator.print("Converting to LoRA")
    lora_config = LoraConfig(
        rank=rank, 
        target_modules=target_modules, 
        exclude_modules=exclude_modules, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        bias=bias, 
        use_rslora=use_rslora
    )

    model = LoraModel(model, lora_config).to(accelerator.device)

accelerator.print(model)

###############################
### Define Training Metrics ###
###############################
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task="multiclass", num_classes=len(labels)).to(accelerator.device)

########################
### Define Optimizer ###
########################
params_to_train = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate, weight_decay=weight_decay)

########################
### DEFINE SCHEDULER ###
########################

scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                            num_warmup_steps=warmup_steps * accelerator.num_processes, 
                                            num_training_steps=epochs * len(trainloader) * accelerator.num_processes)


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

    ### Storage for Everything ###
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    ### Training Progress Bar ###
    progress_bar = tqdm(range(len(trainloader)), disable=not accelerator.is_local_main_process)

    model.train()
    for batch in trainloader:

        ### Move Data to Correct GPU ###
        images, targets = batch["pixel_values"].to(accelerator.device), batch["labels"].to(accelerator.device)
        
        ### Pass Through Model ###
        pred = model(images)["logits"]

        ### Compute and Store Loss ##
        loss = loss_fn(pred, targets)

        ### Compute and Store Accuracy ###
        predicted = pred.argmax(axis=1)
        accuracy = accuracy_fn(predicted, targets)

        ### Compute Gradients ###
        accelerator.backward(loss)

        ### Clip Gradients ###
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        ### Update Model ###
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ### Gather Metrics Across GPUs ###
        loss_gathered = accelerator.gather_for_metrics(loss)
        accuracy_gathered = accelerator.gather_for_metrics(accuracy)

        ### Store Current Iteration Error ###
        train_loss.append(torch.mean(loss_gathered).item())
        train_acc.append(torch.mean(accuracy_gathered).item())
        
        ### Iterate Progress Bar ###
        progress_bar.update(1)

        ### Update Learning Rate ###
        scheduler.step()

    model.eval()
    for batch in tqdm(testloader, disable=not accelerator.is_local_main_process):
        images, targets = batch["pixel_values"].to(accelerator.device), batch["labels"].to(accelerator.device)
        with torch.no_grad():
            pred = model(images)["logits"]

        ### Compute Loss ###
        loss = loss_fn(pred, targets)

        ### Computed Accuracy ###
        predicted = pred.argmax(axis=1)
        accuracy = accuracy_fn(predicted, targets)

        ### Gather across GPUs ###
        loss_gathered = accelerator.gather_for_metrics(loss)
        accuracy_gathered = accelerator.gather_for_metrics(accuracy)

        ### Store Current Iteration Error ###
        test_loss.append(torch.mean(loss_gathered).item())
        test_acc.append(torch.mean(accuracy_gathered).item())
    
    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_test_acc = np.mean(test_acc)

    accelerator.print(f"Training Accuracy: ", epoch_train_acc, "Training Loss:", epoch_train_loss)
    accelerator.print(f"Testing Accuracy: ", epoch_test_acc, "Testing Loss:", epoch_test_loss)
        
    ### Log with Weights and Biases ###
    accelerator.log({"training_loss": epoch_train_loss,
                     "testing_loss": epoch_test_loss, 
                     "training_acc": epoch_train_acc, 
                     "testing_acc": epoch_test_acc}, step=epoch)

### Save Final Model ###
accelerator.wait_for_everyone()

if use_lora:
    accelerator.unwrap_model(model).save_model(os.path.join(working_directory, experiment_name, "food_adapter_checkpoint.safetensors"))
    accelerator.unwrap_model(model).save_model(os.path.join(working_directory, experiment_name, "food_merged_checkpoint.safetensors"), merge_weights=True)
elif not use_lora and train_head_only:
    save_file(accelerator.unwrap_model(model).state_dict(), os.path.join(working_directory, experiment_name, "food_headonly_checkpoint.safetensors"))
else:
    save_file(accelerator.unwrap_model(model).state_dict(), os.path.join(working_directory, experiment_name, "food_fulltrain_checkpoint.safetensors"))

### End Training for Trackers to Exit ###
accelerator.end_training()