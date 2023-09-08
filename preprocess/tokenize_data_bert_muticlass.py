from transformers import DistilBertTokenizerFast
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("./NetCraft")

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the texts in batches to reduce memory usage
def tokenize_function(examples):
    tokenized_examples = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)
    
    # Add the labels to the tokenized examples
    tokenized_examples["labels"] = examples["label"]

    return tokenized_examples



tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_test_dataset = tokenized_dataset.train_test_split(test_size=0.1)
# Split the 10% test + valid in half test, half valid

# Save the dataset
train_test_dataset.save_to_disk('./MutiDebugDistilBERTTok_NetCraft')