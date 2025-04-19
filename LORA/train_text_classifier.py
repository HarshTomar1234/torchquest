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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    DataCollatorWithPadding, get_cosine_schedule_with_warmup

from lora import LoraConfig, LoraModel

import warnings 
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

##########################
### TRAINING ARGUMENTS ###
##########################
experiment_name = "LoRATextClassifier"
wandb_run_name = "bert_lora_classifier"
working_directory = "work_dir"
epochs = 3
batch_size = 64
learning_rate = 3e-5
weight_decay = 0.001
warmup_steps = 100
max_grad_norm = 1.0
num_workers = 32
gradient_checkpointing = True
log_wandb = False
hf_dataset = "imdb"
hf_model_name = "FacebookAI/roberta-base"

######################
### LORA ARGUMENTS ###
######################
use_lora = True
train_head_only = False
target_modules = ["query", "key", "value", "dense", "word_embeddings"]
exclude_modules = ["classifier"] # Dont do LoRA on untrained classifier
rank = 8
lora_alpha = 8
use_rslora = True
bias = "lora_only"
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

########################
### TOKENIZE DATASET ###
########################
dataset = load_dataset(hf_dataset)
labels = dataset["train"].features["label"].names

tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

with accelerator.main_process_first():
    dataset = dataset.map(preprocess_function, batched=True, remove_columns="text") 

###########################
### Prepare DataLoaders ###
###########################

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512)

trainloader = DataLoader(dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True, num_workers=num_workers)
testloader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=data_collator, shuffle=False, num_workers=num_workers)

#######################
### Load LoRA Model ###
#######################
model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, 
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
                    
        ### Pass Through Model ###
        pred = model(**batch)

        ### Grab Loss ###
        loss = pred["loss"]

        ### Compute and Store Accuracy ###
        predicted = pred["logits"].argmax(axis=1)
        accuracy = accuracy_fn(predicted, batch["labels"])

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

        with torch.no_grad():
            pred = model(**batch)

        ### Grab Loss ###
        loss = pred["loss"]

        ### Compute and Store Accuracy ###
        predicted = pred["logits"].argmax(axis=1)
        accuracy = accuracy_fn(predicted, batch["labels"])

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
    accelerator.unwrap_model(model).save_model(os.path.join(working_directory, experiment_name, "imdb_adapter_checkpoint.safetensors"))
    accelerator.unwrap_model(model).save_model(os.path.join(working_directory, experiment_name, "imdb_merged_checkpoint.safetensors"), merge_weights=True)
elif not use_lora and train_head_only:
    save_file(accelerator.unwrap_model(model).state_dict(), os.path.join(working_directory, experiment_name, "imdb_headonly_checkpoint.safetensors"))
else:
    save_file(accelerator.unwrap_model(model).state_dict(), os.path.join(working_directory, experiment_name, "imdb_fulltrain_checkpoint.safetensors"))

### End Training for Trackers to Exit ###
accelerator.end_training()