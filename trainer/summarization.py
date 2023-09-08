from transformers import BartTokenizer, BartForConditionalGeneration
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from utils import *

# sudo python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29522 trainer.py --local_rank -1 
accelerator = Accelerator()
device = accelerator.device

seed = 42
set_seed(seed)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Hyperparameters
batch_size = 16
learning_rate = 5e-5
epochs = 10
max_input_length = 512
max_output_length = 10

# Create dataset and dataloader
dataset = load_from_disk("./data/summarization")
train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset["valid"], batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=True)

valid_length = len(dataset["valid"])
# Define optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
    model, scheduler, optimizer, train_dataloader, valid_dataloader, test_dataloader)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch in tqdm(dataloader):
        input_ids, target_ids = batch["input_ids"], batch["target_ids"]
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=target_ids)
        accelerator.backward(output.loss)

        # Optimization step
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss

            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / valid_length
    print(f"Epoch {epoch + 1} Training Loss: {loss.item()} Validation Loss: {avg_val_loss}")
    
# Save model
model.save_pretrained("./bart_model")