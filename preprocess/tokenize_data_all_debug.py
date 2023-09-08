from transformers import DistilBertTokenizerFast
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("./data/NetCraft")
# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

label2target = {}

with open('./data/label2target_tokencls.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split(maxsplit=1)
        label2target[key] = value

# Tokenize the texts in batches to reduce memory usage
def tokenize_function(examples):
    tokenized_examples = tokenizer(examples["text"].lower(), truncation=True, padding='max_length', max_length=512)
    
    # Add the labels to the tokenized examples
    tokenized_examples["mutilabels"] = examples["label"]
    tokenized_examples["binlabels"] = 0 if examples["label"] == 6 else 1
    tokenized_examples["slabels"] = label2target[str(examples["label"])]
#     qalabels = []
    
    tokenized_examples["stlabels"] = 0
    tokenized_examples["endlabels"] = 0
    
    input_ids = tokenized_examples["input_ids"]
    if examples["label"] != 6:
        labels = []
        target = tokenizer(label2target[str(examples["label"])], truncation=True, padding='do_not_pad')["input_ids"][1:-1]
        length = len(target)
        
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+length] == target:
                labels += [1]*length
                if tokenized_examples["stlabels"] == 0:
                    tokenized_examples["stlabels"] = i
                    tokenized_examples["stlabels"] = i
                    tokenized_examples["endlabels"] = i+length-1
                i += length
            else:
                labels.append(0)
                i += 1
    else:
        labels = [0]*512
    assert len(labels) == 512
    tokenized_examples["tokenlabels"] = labels 
    assert tokenized_examples["slabels"], tokenized_examples["slabels"]
    return tokenized_examples


# tokenized_dataset = dataset.map(tokenize_function, batched=False)
train_test_dataset = dataset.train_test_split(test_size=0.1)
# Split the 10% test + valid in half test, half valid

# Add labels to the tokenized dataset
# tokenized_dataset = tokenized_dataset.add_column('labels', dataset["label"])

# Save the dataset
train_test_dataset.save_to_disk('./data/AllDistilBERTTok_NetCraftv3')