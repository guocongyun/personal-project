import os
import os
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
from transformers import BertConfig, BertModel, BertForSequenceClassification, DistilBertForSequenceClassification, default_data_collator
from datasets import Dataset, load_from_disk, load_metric
from collections import defaultdict
from models.distillbert import DistilBertForToken
from evaluate.strf1 import evaluate
from transformers import DistilBertTokenizerFast
from utils import *
# from seqeval.metrics import f1_score

# sudo python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29522 trainer.py --local_rank -1 
label2target = {}

with open('./data/label2target_tokencls.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split(maxsplit=1)
        label2target[key] = value

accelerator = Accelerator()
device = accelerator.device

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
    configuration.hidden_dropout_prob = dropout
    configuration.attention_probs_dropout_prob = dropout
    if hidden_act not in ("geglu", "reglu"):
        configuration.hidden_act = hidden_act
        model = DistillBertBin(config=configuration)
    else: model = DistillBertBin(config=configuration, custom_hidden_act=hidden_act)
    
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
    
    for epoch in range(start_epoch, epochs):
        model.train()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        preds = defaultdict(lambda: [])
        truth = defaultdict(lambda: [])
        for idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"]
            att_mask = batch["attention_mask"]
            token_labels = batch["tokenlabels"]
            mult_labels = batch["mutilabels"]
            
            if token_labels.sum() == 0: continue
            zero_msk = token_labels.sum(dim=1) != 0
            token_labels = token_labels[zero_msk]
            
            output = model(input_ids=input_ids, attention_mask=att_mask, zero_msk=zero_msk)
            loss = loss_func(output.view(-1, 2), token_labels.view(-1))
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()
            
            preds["train"] += get_keyphrase(input_ids[zero_msk], output)
            truth["train"] += list(map(lambda x: label2target[str(x)], mult_labels[zero_msk].cpu().tolist()))
        print(preds["train"][:50], truth["train"][:50])
        
        model.eval()
        for idx, batch in enumerate(tqdm(valid_dataloader)):
            with torch.no_grad():
                input_ids = batch["input_ids"]
                att_mask = batch["attention_mask"]
                token_labels = batch["tokenlabels"]
                mult_labels = batch["mutilabels"]
                
                if token_labels.sum() == 0: continue
                zero_msk = token_labels.sum(dim=1) != 0
                token_labels = token_labels[zero_msk]
                
                output = model(input_ids=input_ids, attention_mask=att_mask, zero_msk=zero_msk)

                preds["valid"] += get_keyphrase(input_ids[zero_msk], output)
                truth["valid"] += [label2target[str(l)] for l in mult_labels[zero_msk].cpu().tolist()]
#         print(preds["valid"][:50], truth["valid"][:50])
        print(evaluate(preds["train"], truth["train"]))
        print(evaluate(preds["valid"], truth["valid"]))

if __name__ == "__main__":
    trainer(epochs=100, batch_size=64, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, num_hidden_layers=2, file_name=f"new0908mini data 2predrop=0, normal loss")
        
