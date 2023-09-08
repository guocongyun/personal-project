import argparse
import json
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import (
    Adam,
    Adagrad,
    RMSprop,
    SGD,
)
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from collections import defaultdict
from datasets import Dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score, 
    auc, 
    balanced_accuracy_score,
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    DistilBertForSequenceClassification,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    default_data_collator,
)
from models.distillbert import DistilBertForBinMult
from utils import *
# sudo python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29522 trainer.py --local_rank -1 

accelerator = Accelerator()
device = accelerator.device

seed = 42
set_seed(seed)

def trainer(epochs=20,
            batch_size=1,
            data_path="./data/AllDistilBERTTok_NetCraftv2",
            save_path="./new_full",
            pretrained_path=None,
            lr=5e-5,
            weight_decay=1e-3,
            dropout=0.5,
            hidden_act="gelu",
            run_name="Unamed",
            wandb_ids = None,
            optimizer = "AdamW",
            scheduler = "polynomial",
):


    if not os.path.exists(save_path):
        try: os.mkdir(save_path)
        except: pass
    
    configuration = AutoConfig.from_pretrained('distilbert-base-uncased')
    configuration.hidden_dropout_prob = dropout
    configuration.attention_probs_dropout_prob = dropout
    if hidden_act not in ("geglu", "reglu"):
        configuration.hidden_act = hidden_act
        model = DistilBertForBinMult(config=configuration)
    else: model = DistilBertForBinMult(config=configuration, custom_hidden_act=hidden_act)
    
    dataset = load_from_disk(data_path)
    train_dataloader = DataLoader(dataset["train"],collate_fn=default_data_collator, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset["valid"],collate_fn=default_data_collator, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset["test"],collate_fn=default_data_collator, batch_size=batch_size, shuffle=True)
    
    run = wandb.init(id=wandb_ids, project="Personal project", resume="allow")
    if wandb_ids: run=wandb.init(id=wandb_ids, name=run_name, entity="asdf1473", project="safety", resume="allow")
    else: run=wandb.init(name=run_name, entity="asdf1473", project="safety")

    if optimizer == "AdamW": optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adam": optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adagrad": optimizer = Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "RMSprop": optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD": optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "polynomial": scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=(epochs)*len(train_dataloader))
    elif scheduler == "cosine": scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=(epochs)*len(train_dataloader))
    elif scheduler == "linear": scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=(epochs)*len(train_dataloader))

    model, scheduler, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, scheduler, optimizer, train_dataloader, valid_dataloader, test_dataloader)
    
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        torch.cuda.empty_cache()
        train_loss_sum = 0.0
        optimizer.zero_grad()
        preds = defaultdict(lambda: [])
        truth = defaultdict(lambda: [])
        for idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            mult_labels = batch["mutilabels"]
            bin_labels = batch["binlabels"]
            
            bin_output, mult_output = model(input_ids=input_ids, attention_mask=mask)
            mult_loss = loss_func(mult_output, mult_labels)
            bin_loss = loss_func(bin_output, bin_labels)
            combined_loss = (mult_loss + bin_loss)/2
            accelerator.backward(combined_loss)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Move tensors back to cpu for sklearn
            preds["bin_train"] += torch.argmax(bin_output, axis=1).cpu().tolist()
            preds["mult_train"] += torch.argmax(mult_output, axis=1).cpu().tolist()
            truth["bin_train"] += bin_labels.cpu().tolist()
            truth["mult_train"] += mult_labels.cpu().tolist()

        model.eval()
        for idx, batch in enumerate(tqdm(valid_dataloader)):
            with torch.no_grad():
                input_ids = batch["input_ids"]
                mask = batch["attention_mask"]
                mult_labels = batch["mutilabels"]
                bin_labels = batch["binlabels"]
                
                bin_output, mult_output = model(input_ids=input_ids, attention_mask=mask)

                # Move tensors back to cpu for sklearn
                preds["bin_valid"] += torch.argmax(bin_output, axis=1).cpu().tolist()
                preds["mult_valid"] += torch.argmax(mult_output, axis=1).cpu().tolist()
                truth["bin_valid"] += bin_labels.cpu().tolist()
                truth["mult_valid"] += mult_labels.cpu().tolist()
            
        bin_train_f1 = f1_score(truth["bin_train"], preds["bin_train"], average='weighted')  
        mult_train_f1 = f1_score(truth["mult_train"], preds["mult_train"], average='weighted')  
        bin_valid_f1 = f1_score(truth["bin_valid"], preds["bin_valid"], average='weighted')  
        mult_valid_f1 = f1_score(truth["mult_valid"], preds["mult_valid"], average='weighted')  
        print('F1 Score:', bin_train_f1, mult_train_f1)
        print('F1 Score:', bin_valid_f1, mult_valid_f1)
        name = run_name+f"epoch_{epoch}.pth"
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_path, name))
            
#         print(classification_report(val_true2, val_pred2))
#         wandb.log({"epoch": epoch, 
#                    "loss": train_loss_sum/len(train_dataloader),
# #                        "roc_train":roc_auc_score(val_true3, val_score3), 
#                    "acc_train":accuracy_score(val_true3, val_pred3),
#                    "balanced_acc_train":balanced_accuracy_score(val_true3, val_pred3),
# #                        "f1_score_train":f1_score(val_true3, val_pred3),
# #                        "roc_vali":roc_auc_score(val_true1, val_score1), 
# #                        "acc_vali":accuracy_score(val_true1, val_pred1),
# #                        "f1_score_vali":f1_score(val_true1, val_pred1),
# #                        "roc_test":roc_auc_score(val_true2, val_score2), 
#                    "acc_test":accuracy_score(val_true2, val_pred2),
#                    "balanced_acc_test":balanced_accuracy_score(val_true2, val_pred2),
# #                        "f1_score_test":f1_score(val_true2, val_pred2),
# #                        "roc_online":roc_auc_score(val_true1, val_score1), 
#                    "acc_online":accuracy_score(val_true1, val_pred1),
#                    "balanced_acc_online":balanced_accuracy_score(val_true1, val_pred1),
# #                        "f1_score_online":f1_score(val_true1, val_pred1),
#             })

if __name__ == "__main__":
    trainer(epochs=100, batch_size=64, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, num_hidden_layers=2, file_name=f"new0908mini data 2predrop=0, normal loss")
        
