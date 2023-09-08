from transformers import DistilBertTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
from collections import defaultdict
import pickle

# Load the dataset
# dataset = load_from_disk("./data/AllDistilBERTTok_NetCraftv2")
dataset = load_from_disk("./data/NetCraft")

def count(dataset):
    count = defaultdict(lambda: 0)
    for d in tqdm(dataset):
        count[d["label"]] += 1
    count = sorted(count.items(), key=lambda item: item[1], reverse=True)
    return count

label2target = {}

with open('./data/label2target_tokencls.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split(maxsplit=1)
        label2target[key] = value

cts = count(dataset)
n = len(cts)
keys = [[] for _ in range(3)]
counts = [0]*3
for i, c in enumerate(cts):
    k = i % 10
    if k == 3:
        keys[1].append(c[0])
        counts[1] += c[1]
    elif k == 2:
        keys[2].append(c[0])
        counts[2] += c[1]
    else:
        keys[0].append(c[0])
        counts[0] += c[1]


with open('./data/unk_keys.pkl', 'wb') as file:
    pickle.dump(keys, file)
        
print(counts)
for k in keys:
    print(len(k))
for k in keys[2]:
    print(label2target[str(k)])
# print(keys)
# print(n)
# print(sum([len(k) for k in keys]))    
    
