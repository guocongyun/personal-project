import os
import pickle
import time
import json
import numpy as np
import random
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AdamW, \
    get_cosine_schedule_with_warmup, \
    get_linear_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, classification_report, roc_curve, auc, f1_score, recall_score, precision_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import BertConfig, BertModel, BertForSequenceClassification, DistilBertForSequenceClassification, default_data_collator, AutoConfig
from datasets import Dataset, load_from_disk
from sklearn.metrics import f1_score
from collections import defaultdict
from models.distillbert import DistilBertForBinMultToken
from trainer.tokenclass import get_keyphrase
from evaluate.strf1 import evaluate
from transformers import DistilBertTokenizerFast
from utils import *
# sudo python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29522 trainer.py --local_rank -1 

accelerator = Accelerator()
device = accelerator.device

label2target = {}

with open('./data/label2target_tokencls.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split(maxsplit=1)
        label2target[key] = value

seed = 42
set_seed(seed)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

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
#     configuration.hidden_dropout_prob = dropout
#     configuration.attention_probs_dropout_prob = dropout
    if hidden_act not in ("geglu", "reglu"):
#         configuration.hidden_act = hidden_act
        model = DistilBertForBinMultToken(config=configuration)
    else: model = DistilBertForBinMultToken(config=configuration, custom_hidden_act=hidden_act)
    
    dataset = load_from_disk(data_path)
    train_dataloader = DataLoader(dataset["train"],collate_fn=default_data_collator, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset["valid"],collate_fn=default_data_collator, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset["test"],collate_fn=default_data_collator, batch_size=batch_size, shuffle=True)
    
#     run = wandb.init(id=wandb_ids, project="Personal project", resume="allow")
#     if wandb_ids: run=wandb.init(id=wandb_ids, name=run_name, entity="asdf1473", project="safety", resume="allow")
#     else: run=wandb.init(name=run_name, entity="asdf1473", project="safety")

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
    
    with open('./data/unk_keys.pkl', 'rb') as file:
        keys = pickle.load(file)
        keys = [torch.tensor(key, device=device) for key in keys]
    
    for epoch in range(epochs):
        model.train()
        torch.cuda.empty_cache()
        train_loss_sum = 0.0
        optimizer.zero_grad()
        preds = defaultdict(lambda: [])
        truth = defaultdict(lambda: [])
        for idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"]
            att_mask = batch["attention_mask"]
            mult_labels = batch["mutilabels"]
            bin_labels = batch["binlabels"]
            token_labels = batch["tokenlabels"]
            
#             benign_mask = batch["mutilabels"] !=  6
#             input_ids = input_ids[benign_mask]
#             att_mask = att_mask[benign_mask]
#             mult_labels = mult_labels[benign_mask]
#             bin_labels = bin_labels[benign_mask]
#             token_labels = token_labels[benign_mask]
            
            test_mask = torch.isin(mult_labels, keys[1])
            input_ids = input_ids[~test_mask]
            att_mask = att_mask[~test_mask]
            mult_labels = mult_labels[~test_mask]
            bin_labels = bin_labels[~test_mask]
            token_labels = token_labels[~test_mask]
            
            zero_msk = token_labels.sum(dim=1) != 0
            token_labels = token_labels[zero_msk]
            token_train = [label2target[str(l)] for l in mult_labels[zero_msk].cpu().tolist()]
            
            unk_mask = torch.isin(mult_labels, keys[2])
            mult_labels = torch.where(unk_mask, torch.tensor(0), mult_labels)
            
            if token_labels.sum() == 0: continue
            bin_output, mult_output, token_output = model(input_ids=input_ids, attention_mask=att_mask, zero_msk=zero_msk)
            bin_loss = loss_func(bin_output, bin_labels)
            mult_loss = loss_func(mult_output, mult_labels)
            token_loss = loss_func(token_output.view(-1, 2), token_labels.view(-1))
            combined_loss = (mult_loss + bin_loss + token_loss)/3
            accelerator.backward(combined_loss)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
#             preds["input_ids"] += input_ids.cpu().tolist()
            preds["bin_train"] += torch.argmax(bin_output, axis=1).cpu().tolist()
            preds["mult_train"] += torch.argmax(mult_output, axis=1).cpu().tolist()
            preds["token_train"] += get_keyphrase(input_ids[zero_msk], token_output)
            
            truth["bin_train"] += bin_labels.cpu().tolist()
            truth["mult_train"] += mult_labels.cpu().tolist()
            truth["token_train"] += token_train

        model.eval()
        for idx, batch in enumerate(tqdm(valid_dataloader)):
            with torch.no_grad():
                input_ids = batch["input_ids"]
                att_mask = batch["attention_mask"]
                mult_labels = batch["mutilabels"]
                bin_labels = batch["binlabels"]
                token_labels = batch["tokenlabels"]
                
#                 benign_mask = batch["mutilabels"] !=  6
#                 input_ids = input_ids[benign_mask]
#                 att_mask = att_mask[benign_mask]
#                 mult_labels = mult_labels[benign_mask]
#                 bin_labels = bin_labels[benign_mask]
#                 token_labels = token_labels[benign_mask]
                
                unk_mask = torch.isin(mult_labels, keys[1])
                
                input_ids = input_ids[unk_mask]
                att_mask = att_mask[unk_mask]
                mult_labels = mult_labels[unk_mask]
                bin_labels = bin_labels[unk_mask]
                token_labels = token_labels[unk_mask]
                
                zero_msk = token_labels.sum(dim=1) != 0
                
                token_valid = [label2target[str(l)] for l in mult_labels[zero_msk].cpu().tolist()]
#                 mult_labels = torch.where(unk_mask, torch.tensor(0), mult_labels)
                
                if token_labels.sum() == 0: continue
                bin_output, mult_output, token_output = model(input_ids=input_ids, attention_mask=att_mask, zero_msk=zero_msk)

                # Move tensors back to cpu for sklearn
                preds["input_ids"] += input_ids.cpu().tolist()
                preds["bin_valid"] += torch.argmax(bin_output, axis=1).cpu().tolist()
                preds["mult_valid"] += torch.argmax(mult_output, axis=1).cpu().tolist()
                preds["token_valid"] += get_keyphrase(input_ids[zero_msk], token_output)
                
                truth["bin_valid"] += bin_labels.cpu().tolist()
                truth["mult_valid"] += mult_labels.cpu().tolist()
                truth["token_valid"] += token_valid
        k = len(preds["input_ids"])
        counter = defaultdict(lambda:0)
        
        for i in range(k):
            if preds["bin_valid"][i] != truth["bin_valid"][i]:
                counter[truth["mult_valid"][i]] += 1
        counter = sorted(counter.items(), key=lambda item: item[1], reverse=True)
        
        print(counter)
        
#                 print(tokenizer.decode(preds["input_ids"][i]))
                
        bin_train_f1 = f1_score(truth["bin_train"], preds["bin_train"], average='weighted')  
        mult_train_f1 = f1_score(truth["mult_train"], preds["mult_train"], average='weighted')  
        token_train_f1 = evaluate(preds["token_train"], truth["token_train"])
        print('F1 Score:', bin_train_f1, mult_train_f1, token_train_f1)
        
        bin_valid_f1 = f1_score(truth["bin_valid"], preds["bin_valid"], average='weighted')  
        mult_valid_f1 = f1_score(truth["mult_valid"], preds["mult_valid"], average='weighted')  
        token_valid_f1 = evaluate(preds["token_valid"], truth["token_valid"])
        print('F1 Score:', bin_valid_f1, mult_valid_f1, token_valid_f1)

if __name__ == "__main__":
    trainer(epochs=100, batch_size=64)
        
