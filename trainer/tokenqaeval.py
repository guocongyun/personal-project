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
from utils import *
# sudo python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29522 trainer.py --local_rank -1 

label2target = {}

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
    
    nlp = pipeline('question-answering', model="deepset/tinyroberta-squad2", tokenizer="deepset/tinyroberta-squad2")

    dataset = load_from_disk(data_path)
    preds = defaultdict(lambda: [])
    truth = defaultdict(lambda: [])
    for idx, data in enumerate(tqdm(dataset["valid"])):
        if not sum(data["tokenlabels"]): continue
        QA_input = {
            'question': 'Which company is this email from?',
            'context': data["text"]
        }
        res = nlp(QA_input)
#         print(type(res), res)
    # b) Load model & tokenizer
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

#     run = wandb.init(id=wandb_ids, project="Personal project", resume="allow")
#     if wandb_ids: run=wandb.init(id=wandb_ids, name=run_name, entity="asdf1473", project="safety", resume="allow")
#     else: run=wandb.init(name=run_name, entity="asdf1473", project="safety")

        preds["token_valid"] += [res['answer']]
        truth["token_valid"] += [label2target[str(data["mutilabels"])]]
        if idx % 100 == 1:
            token_valid_f1 = evaluate(preds["token_valid"], truth["token_valid"])
            print('F1 Score:', token_valid_f1)

if __name__ == "__main__":
    trainer(epochs=100, batch_size=64)
        
