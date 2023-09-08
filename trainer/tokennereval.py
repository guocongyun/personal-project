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
from transformers import BertConfig, BertModel, BertForSequenceClassification, DistilBertForSequenceClassification, default_data_collator, AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from datasets import Dataset, load_from_disk
from sklearn.metrics import f1_score
from collections import defaultdict
from models.distillbert import DistilBertForBinMultToken
from trainer.tokenclass import get_keyphrase
from evaluate.strf1 import evaluate
from transformers import DistilBertTokenizerFast
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from utils import *
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
    

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    
    dataset = load_from_disk(data_path)
    preds = defaultdict(lambda: [])
    truth = defaultdict(lambda: [])
    for idx, data in enumerate(tqdm(dataset["valid"])):
        if not sum(data["tokenlabels"]): continue
        predictions = nlp(data["text"])
#         print(res)
#         print(data["text"])
#         break
        # Initialize variables to hold current organization and frequency dictionary
        current_org = []
        org_freq = defaultdict(lambda: 0)
        org_freq[""] = 0
        # Iterate through the predictions
        for pred in predictions:
            if pred['entity'] == 'B-ORG':
                # If a new organization is found, join the previous words together and add to the frequency dictionary
                if current_org:
                    org_name = ' '.join(current_org).replace(' ##', '')
                    org_freq[org_name] += 1
                    current_org = []
                current_org.append(pred['word'])
            elif pred['entity'] == 'I-ORG' and current_org:
                # If a continuation of the organization is found, add the word to the current organization
                current_org.append(pred['word'])

        # Add the last organization if there is one
        if current_org:
            org_name = ' '.join(current_org).replace(' ##', '')
            org_freq[org_name] += 1
        pred = max(org_freq, key=org_freq.get)
#         print(type(res), res)
    # b) Load model & tokenizer
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

#     run = wandb.init(id=wandb_ids, project="Personal project", resume="allow")
#     if wandb_ids: run=wandb.init(id=wandb_ids, name=run_name, entity="asdf1473", project="safety", resume="allow")
#     else: run=wandb.init(name=run_name, entity="asdf1473", project="safety")

        preds["token_valid"] += [pred]
        truth["token_valid"] += [label2target[str(data["mutilabels"])]]
        if idx % 100 == 1:
            token_valid_f1 = evaluate(preds["token_valid"], truth["token_valid"])
            print('F1 Score:', token_valid_f1)

if __name__ == "__main__":
    trainer(epochs=100, batch_size=64)
        
