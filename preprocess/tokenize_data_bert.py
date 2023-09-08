from transformers import DistilBertTokenizerFast
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("./NetCraft_test")

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the texts in batches to reduce memory usage
def tokenize_function(examples):
    tokenized_examples = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)
    
    # Convert all non-zero labels to 1
    labels = [0 if label == 6 else 1 for label in examples["label"]]
    
    # Add the labels to the tokenized examples
    tokenized_examples["labels"] = labels

    return tokenized_examples



tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Add labels to the tokenized dataset
# tokenized_dataset = tokenized_dataset.add_column('labels', dataset["label"])

# Save the dataset
tokenized_dataset.save_to_disk('./DistilBERTTok_NetCraft')