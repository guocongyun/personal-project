from transformers import DistilBertTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
from collections import defaultdict


# Load the dataset
# dataset = load_from_disk("./DistilBERTTok_NetCraft")
dataset = load_from_disk("./data/NetCraft")
# dataset = load_from_disk("./DebugSplitDistilBERTTok_NetCraft")

def count(dataset):
    count = defaultdict(lambda: 0)
    k = 0
    for d in tqdm(dataset):
        print(d)
        print(d.keys())
        k += 1
        if k > 20: break
        count[d["label"]] += 1
    count = sorted(count.items(), key=lambda item: item[1], reverse=True)
    return count


cts = count(dataset)
n = len(cts)
print(cts)
keys = [[] for _ in range(3)]
counts = [0]*3
for i, c in enumerate(cts):
    if i == 0: continue
    k = i % 10
    if k in [2, 3] and counts[1] < counts[2]:
        keys[1].append(c[0])
        counts[1] += c[1]
    elif k in [2, 3] and counts[1] >= counts[2]:
        keys[2].append(c[0])
        counts[2] += c[1]
    else:
        keys[0].append(c[0])
        counts[0] += c[1]
        
print(counts)
for k in keys:
    print(len(k))
print(keys)
print(n)
print(sum([len(k) for k in keys]))    
    
