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
from datasets import Dataset, load_from_disk
from sklearn.metrics import f1_score
from collections import defaultdict
from models.distillbert import DistilBertForBinMultTokenKNN
from trainer.tokenclass import get_keyphrase
from evaluate.strf1 import evaluate
from datasets import concatenate_datasets
from trainer.binary import trainer
from utils import *
# sudo python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29522 trainer.py --local_rank -1 

# {
#     "data_path": "./data/AllDistilBERTTok_NetCraftv2",
#     "pretrained_path": "./pretrain/distillbert",
#     "save_path": "./distillbert",
#     "run_name": "distillbert",
#     "epochs": 20,
#     "dropout": 0.5,
#     "weight decay": 1e-3,
#     "batch_size": 64,
#     "lr": 5e-5,
#     "optimizer": "AdamW",
#     "scheduler": "polynomial",
#     "hidden_act": "relu",
#     "wandb_ids": 132,
# }

if __name__ == "__main__":
    
    # with open('./configs/distillbert.json', 'r') as file:
    #     config = json.load(file)
    # set_seed(config["seed"])
    # default_config = config.copy()
    # for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
    #     config = default_config.copy()
    #     config["lr"] = lr
    #     config["run_name"] = f"distillbert_lr_{lr}"
    #     trainer(**config)
        
    # for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
    #     config = default_config.copy()
    #     config["weight_decay"] = weight_decay
    #     config["run_name"] = f"distillbert_weight_decay_{weight_decay}"
    #     trainer(**config)
        
    # for dropout in [0.1, 0.3, 0.5]:
    #     config = default_config.copy()
    #     config["dropout"] = dropout
    #     config["run_name"] = f"distillbert_dropout_{dropout}"
    #     trainer(**config)
        
    # for optimizer in ["AdamW", "Adam", "Adagrad", "RMSprop", "SGD"]:
    #     config = default_config.copy()
    #     config["optimizer"] = optimizer
    #     config["run_name"] = f"distillbert_optimizer_{optimizer}"
    #     trainer(**config)
        
    # for scheduler in ["polynomial", "cosine", "linear"]:
    #     config = default_config.copy()
    #     config["scheduler"] = scheduler
    #     config["run_name"] = f"distillbert_scheduler_{scheduler}"
    #     trainer(**config)
        
    # for hidden_act in ["reglu", "geglu", "relu", "silu", "gelu"]:
    #     config = default_config.copy()
    #     config["hidden_act"] = hidden_act
    #     config["run_name"] = f"distillbert_hidden_act_{hidden_act}"
        trainer(**config)
        
    with open('./configs/bert.json', 'r') as file:
        config = json.load(file)
    set_seed(config["seed"])
    default_config = config.copy()
    for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
        config = default_config.copy()
        config["lr"] = lr
        config["run_name"] = f"distillbert_lr_{lr}"
        trainer(**config)
        
    for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
        config = default_config.copy()
        config["weight_decay"] = weight_decay
        config["run_name"] = f"bert_weight_decay_{weight_decay}"
        trainer(**config)
        
    for dropout in [0.1, 0.3, 0.5]:
        config = default_config.copy()
        config["dropout"] = dropout
        config["run_name"] = f"bert_dropout_{dropout}"
        trainer(**config)
        
    for optimizer in ["AdamW", "Adam", "Adagrad", "RMSprop", "SGD"]:
        config = default_config.copy()
        config["optimizer"] = optimizer
        config["run_name"] = f"bert_optimizer_{optimizer}"
        trainer(**config)
        
    for scheduler in ["polynomial", "cosine", "linear"]:
        config = default_config.copy()
        config["scheduler"] = scheduler
        config["run_name"] = f"bert_scheduler_{scheduler}"
        trainer(**config)
        
    for hidden_act in ["reglu", "geglu", "relu", "silu", "gelu"]:
        config = default_config.copy()
        config["hidden_act"] = hidden_act
        config["run_name"] = f"bert_hidden_act_{hidden_act}"
        trainer(**config)
        
    with open('./configs/albert.json', 'r') as file:
        config = json.load(file)
    set_seed(config["seed"])
    default_config = config.copy()
    for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
        config = default_config.copy()
        config["lr"] = lr
        config["run_name"] = f"albert_lr_{lr}"
        trainer(**config)
        
    for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
        config = default_config.copy()
        config["weight_decay"] = weight_decay
        config["run_name"] = f"albert_weight_decay_{weight_decay}"
        trainer(**config)
        
    for dropout in [0.1, 0.3, 0.5]:
        config = default_config.copy()
        config["dropout"] = dropout
        config["run_name"] = f"albert_dropout_{dropout}"
        trainer(**config)
        
    for optimizer in ["AdamW", "Adam", "Adagrad", "RMSprop", "SGD"]:
        config = default_config.copy()
        config["optimizer"] = optimizer
        config["run_name"] = f"albert_optimizer_{optimizer}"
        trainer(**config)
        
    for scheduler in ["polynomial", "cosine", "linear"]:
        config = default_config.copy()
        config["scheduler"] = scheduler
        config["run_name"] = f"albert_scheduler_{scheduler}"
        trainer(**config)
        
    for hidden_act in ["reglu", "geglu", "relu", "silu", "gelu"]:
        config = default_config.copy()
        config["hidden_act"] = hidden_act
        config["run_name"] = f"albert_hidden_act_{hidden_act}"
        trainer(**config)

    with open('./configs/rnn.json', 'r') as file:
        config = json.load(file)
    set_seed(config["seed"])
    default_config = config.copy()
    for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
        config = default_config.copy()
        config["lr"] = lr
        config["run_name"] = f"distillbert_lr_{lr}"
        trainer(**config)
        
    for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
        config = default_config.copy()
        config["weight_decay"] = weight_decay
        config["run_name"] = f"bert_weight_decay_{weight_decay}"
        trainer(**config)
        
    for dropout in [0.1, 0.3, 0.5]:
        config = default_config.copy()
        config["dropout"] = dropout
        config["run_name"] = f"bert_dropout_{dropout}"
        trainer(**config)
        
    for optimizer in ["AdamW", "Adam", "Adagrad", "RMSprop", "SGD"]:
        config = default_config.copy()
        config["optimizer"] = optimizer
        config["run_name"] = f"bert_optimizer_{optimizer}"
        trainer(**config)
        
    for scheduler in ["polynomial", "cosine", "linear"]:
        config = default_config.copy()
        config["scheduler"] = scheduler
        config["run_name"] = f"bert_scheduler_{scheduler}"
        trainer(**config)
        
    for hidden_act in ["reglu", "geglu", "relu", "silu", "gelu"]:
        config = default_config.copy()
        config["hidden_act"] = hidden_act
        config["run_name"] = f"bert_hidden_act_{hidden_act}"
        trainer(**config)

    with open('./configs/cnn.json', 'r') as file:
        config = json.load(file)
    set_seed(config["seed"])
    default_config = config.copy()
    for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
        config = default_config.copy()
        config["lr"] = lr
        config["run_name"] = f"distillbert_lr_{lr}"
        trainer(**config)
        
    for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
        config = default_config.copy()
        config["weight_decay"] = weight_decay
        config["run_name"] = f"bert_weight_decay_{weight_decay}"
        trainer(**config)
        
    for dropout in [0.1, 0.3, 0.5]:
        config = default_config.copy()
        config["dropout"] = dropout
        config["run_name"] = f"bert_dropout_{dropout}"
        trainer(**config)
        
    for optimizer in ["AdamW", "Adam", "Adagrad", "RMSprop", "SGD"]:
        config = default_config.copy()
        config["optimizer"] = optimizer
        config["run_name"] = f"bert_optimizer_{optimizer}"
        trainer(**config)
        
    for scheduler in ["polynomial", "cosine", "linear"]:
        config = default_config.copy()
        config["scheduler"] = scheduler
        config["run_name"] = f"bert_scheduler_{scheduler}"
        trainer(**config)
        
    for hidden_act in ["reglu", "geglu", "relu", "silu", "gelu"]:
        config = default_config.copy()
        config["hidden_act"] = hidden_act
        config["run_name"] = f"bert_hidden_act_{hidden_act}"
        trainer(**config)
        