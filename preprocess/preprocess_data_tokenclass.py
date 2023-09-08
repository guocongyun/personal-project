from transformers import DistilBertTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
from collections import defaultdict

# Load the dataset
dataset = load_from_disk("./NetCraft")

label2target = {}

with open('label2target_tokencls.txt', 'r') as f:
    for line in f:
#         print(line.strip().split(maxsplit=1))
        key, value = line.strip().split(maxsplit=1)
        label2target[key] = value
# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# print(tokenizer("dhl"))
# print(tokenizer("DHL"))
# print(tokenizer.decode([5653]))
# print(tokenizer.decode([1050, 15290, 24475]))
# exit()
match = 0
not_matched = defaultdict(lambda: 0)
for d in tqdm(dataset):
    labels = []
    matched = False
    if d["label"] != 6:
        tokenized_examples = tokenizer(d["text"].lower(), truncation=True, padding='max_length', max_length=512)["input_ids"]
        target = tokenizer(label2target[str(d["label"])], truncation=True, padding='do_not_pad')["input_ids"][1:-1]
        
        i = 0
        length = len(target)
        while i < len(tokenized_examples):
            if tokenized_examples[i:i+length] == target:
                labels += [1] * length
                i += length
                match += 1
                matched = True
#                 break
            else:
                labels.append(0)
                i += 1
    else:
        labels = [0]*512
    if not matched and d["label"] != 6:
        not_matched[label2target[str(d["label"])]] += 1
#         print(label2target[str(d["label"])], "--------", d["text"])
#         print("----------------------------------------------------------------------")
    assert len(labels) == 512, [len(labels), target, len(tokenized_examples), length, i]
tup = sorted(not_matched.items(), key=lambda item: item[1], reverse=True)
print(tup)
print(match)
exit()
        
#         exit()
#     if label2target[d["label"]] in 


# Tokenize the texts in batches to reduce memory usage
# def tokenize_function(examples):
#     tokenized_examples = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)
    
#     # Add the labels to the tokenized examples
#     tokenized_examples["labels"] = examples["label"]

#     return tokenized_examples



tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_test_dataset = tokenized_dataset.train_test_split(test_size=0.1)
# Split the 10% test + valid in half test, half valid
# Add labels to the tokenized dataset
# tokenized_dataset = tokenized_dataset.add_column('labels', dataset["label"])

# Save the dataset
train_test_dataset.save_to_disk('./MutiDebugDistilBERTTok_NetCraft')