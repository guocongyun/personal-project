from transformers import BertTokenizerFast
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("./Tokenize_NetCraft")

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the texts in batches to reduce memory usage
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Add labels to the tokenized dataset
tokenized_dataset = tokenized_dataset.add_column('labels', dataset["label"])

# Save the dataset
tokenized_dataset.save_to_disk('./Tokenize_BinNetCraft')