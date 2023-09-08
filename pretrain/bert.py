from transformers import BertTokenizer, BertForMaskedLM, BertModel, DataCollatorForLanguageModeling, AutoConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import random
from datasets import load_from_disk
from models.distillbert import BertForPretrain
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import concatenate_datasets

accelerator = Accelerator()
device = accelerator.device

dataset = load_from_disk("./data/NetCraft")
train_data = dataset["train"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def pretrain():
    
    # DataLoader
    train_dataloader = DataLoader(train_data, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15), batch_size=32, shuffle=True)
    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = BertForPretrain(config).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    # Training
    model.train()
    for epoch in range(5):  # Number of epochs
        for batch in train_dataloader:
            # Move batch to GPU
            batch = {k: v.to('cuda') for k, v in batch.items()}

            # Forward pass
            mlm_outputs, sop_outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], sop_labels=batch['sop_labels'])
            loss = outputs[0]
            # Compute loss

            accelerator.backward(combined_loss)

            optimizer.step()
            optimizer.zero_grad()

            print(f"Loss: {loss.item()}")

    # Save model
    model_save_path = "./saves/distillbert_pretrain"
    model.bert.save_pretrained(model_save_path)

if __name__ == "__main__":
    pretrain()