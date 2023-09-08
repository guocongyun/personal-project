from transformers import DistilBertTokenizerFast
from datasets import load_from_disk, Dataset

# Load the dataset
dataset = load_from_disk("./DistilBERTTok_NetCraft")

# 90% train, 10% test + validation
train_test_dataset = dataset.train_test_split(test_size=0.1)
# Split the 10% test + valid in half test, half valid

# Save the dataset
train_test_dataset.save_to_disk('./DebugDistilBERTTok_NetCraft')