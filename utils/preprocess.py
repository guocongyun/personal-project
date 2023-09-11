import tensorflow as tf 
from tqdm import tqdm
import email
import re
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from datasets import Dataset
import matplotlib.pyplot as plt
# train_key = [6, 13, 210, 39, 9, 42, 23, 245, 22, 74, 30, 55, 83, 142, 41, 355, 31, 132, 181, 18, 279, 52, 33, 420, 51, 196, 5, 329, 174, 349, 217, 35, 80, 188, 179, 288, 397, 388, 118, 180, 60, 117, 158, 86, 96, 81, 152, 11, 67, 213, 171, 271, 462, 136, 115, 403, 391, 72, 240, 231, 73, 435, 148, 32, 208, 299, 233, 49, 296, 58, 85, 273, 21, 17, 202, 194, 97, 353, 328, 129, 168, 321, 451, 166, 146, 124, 140, 419, 251, 78, 287, 53, 120, 14, 46, 24, 317, 305, 16, 220, 54, 258, 216, 34, 309, 212, 373, 303, 457, 266, 126, 274, 408, 333, 82, 63, 228, 342, 176, 239, 244, 44, 385, 302, 334, 402, 56, 71, 322, 221, 134, 340, 361, 157, 121, 38, 445, 362, 47, 378, 382, 343, 144, 346, 398, 95, 135, 311, 284, 15, 99, 238, 358, 156, 250, 204, 192, 407, 438, 356, 248, 91, 125, 253, 387, 226, 453, 184, 198, 70, 307, 297, 278, 344, 295, 153, 268, 141, 27, 428, 243, 401, 218, 182, 400, 241, 374, 392, 330, 285, 48, 183, 276, 277, 203, 122, 224, 371, 201, 426, 175, 394, 89, 37, 289, 66, 393, 149, 222, 229, 79, 347, 256, 264, 372, 352, 61, 383, 318, 75, 320, 257, 357, 128, 327, 337, 364, 447, 170, 434, 209, 316, 225, 312, 93, 310, 237, 191, 232, 304, 260, 324, 252, 325, 326, 161, 234, 370, 425, 167, 150, 137, 200, 214, 211, 431, 262, 185, 412, 341, 173, 406, 449, 235, 36, 28, 119, 290, 359, 348, 345, 375, 145, 43, 255, 339, 94, 442, 293, 205, 294, 314, 446, 313, 283, 331, 236, 68, 376, 300, 459, 223, 127, 429, 143, 187, 381, 338, 368, 164, 432, 360, 10, 367, 335, 443, 165, 308, 298, 433, 315, 263, 396, 354, 282, 178, 415, 172, 332, 389, 130, 386, 84, 395, 59, 416, 439, 440, 215, 246, 219, 323, 281, 76, 404, 227, 380, 369, 377, 291, 444, 275, 441, 261, 206, 230, 418, 249, 270, 254, 460, 12, 411, 111, 19, 197, 363, 269, 427, 272, 247, 458, 455, 390, 155, 424, 450, 154, 365, 410, 461, 114, 413, 384, 301, 138, 98, 195, 265, 422, 259, 26, 409, 186, 199, 336, 423]
# test_key = [6, 39, 24, 9, 210, 171, 58, 30, 13, 258, 148, 41, 67, 31, 420, 349, 74, 117, 451, 274, 42, 55, 23, 5, 86, 212, 49, 80, 248, 32, 273, 403, 22, 279, 217, 129, 11, 295, 115, 142, 202, 118, 85, 271, 361, 179, 93, 152, 136, 35, 83, 53, 308, 321, 216, 158, 348, 240, 329, 166, 256, 228, 233, 73, 150, 232, 168, 268, 231, 438, 287, 17, 245, 44, 96, 355, 91, 334, 180, 75, 188, 52, 60, 72, 174, 124, 135, 97, 408, 54, 330, 387, 462, 430, 78, 437, 391, 297, 70, 445, 385, 204, 390, 457, 458, 34, 33, 46, 203, 322, 373, 82, 388, 132, 251, 144, 327, 89, 378, 441, 140, 309, 317, 183, 364, 293, 319, 306, 236, 146, 176, 47, 120, 394, 18, 201, 181, 323, 29, 425, 367, 194, 333, 51, 338, 358, 342, 401, 126, 340, 153, 235, 398, 356, 294, 371, 288, 296, 386, 99, 436, 56, 244, 359, 351, 243, 14, 421, 229, 379, 400, 362, 167, 218, 27, 427, 343, 267, 307, 211, 200, 149, 81, 21, 122, 221, 111, 435, 68, 241, 302, 352, 230, 157, 402, 213, 59, 61, 154, 196, 285, 382, 195, 357, 38, 374, 15, 314, 292, 405, 224, 182, 20, 214, 175, 318, 170, 454, 372, 252, 366, 266, 399, 280, 277, 270, 239, 278, 275, 66, 381, 95, 79, 125, 350, 303, 456, 209, 455, 264, 222, 447, 161, 396, 410, 130, 368, 298, 100, 128, 16, 134, 137, 414, 353, 242, 389, 250, 443, 71, 286, 185, 127, 143, 141, 198, 220, 301, 429, 238, 237, 452, 413, 257, 191, 417, 439, 407, 208, 121, 207, 448, 406, 162]
# ha = 0
# ho = 0
# for k in train_key:
#     if k not in test_key:
#         ha += 1
# for k in test_key:
#     if k not in train_key:
#         ho += 1
# print(len(train_key))
# print(ha)
# print(ho)

# exit()

def read_mappings(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        mappings = {}
        for line in lines:
            parts = line.strip().split()
            mappings[int(parts[0])] = parts[1]
        return mappings

train_dataset = tf.data.TFRecordDataset("./data_tfrecord_train/data_tfrecord_train")
test_dataset = tf.data.TFRecordDataset("./data_tfrecord_test/data_tfrecord_test")


ids = []
with open("./mappings") as f:
    lines = f.readlines()
    lines = [l.strip().split() for l in lines[1:]]
    ids = set([int(idx) for idx, name in lines])

with open("./mappings") as f:
    lines = f.readlines()
    mappings = {}
    for line in lines[1:]:
        parts = line.strip().split()
        mappings[int(parts[0])] = parts[1]

texts = []
# labels = []
labels_count = defaultdict(int)  # Initializing dictionary to count labels

for raw_dataset in [train_dataset, test_dataset]:
    lengths = []
    # labels = defaultdict(lambda:0)
    for idx, r in tqdm(enumerate(raw_dataset)):
        example = tf.train.Example()
        example.ParseFromString(r.numpy())

        # example.ParseFromString(r.numpy())
        # body = example.features.feature["body"].bytes_list.value[0]
        label = example.features.feature["label_id"].int64_list.value[0]
        labels_count[label] += 1
        # labels[label] += 1

top_8_labels = dict(sorted(labels_count.items(), key=lambda item: item[1], reverse=True)[:10])
label_names = [mappings[label] for label in top_8_labels.keys()]
# Plotting
plt.figure(figsize=(12, 7))
plt.bar(label_names, top_8_labels.values(), color='black')
plt.xlabel('Phishing target')
plt.ylabel('Count')
plt.title('Distribution of samples in Netcraft dataset')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent overlap
print(top_8_labels)
plt.savefig('Datadist.png')  # Choose a suitable name and format for your figure
print(sum(list(labels_count.values())))
plt.show()

        # soup = BeautifulSoup(body, "lxml")
        # for script in soup(["class", "id", "name","script", "style"]):
        #     script.extract()  
        # text = soup.get_text()
        # lines = (line.strip() for line in text.splitlines())
        # chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # if chunks and chunks[0] == 'This is a multi-part message in MIME format.':
        #     chunks = chunks[1:]
        # counts = Counter(chunks)
        # chunks = [item for item, count in counts.items() if count == 1 and item[:13] != "Content-Type:" ]
        # k = len(chunks)
        # if k <= 5: continue
        # text = '\n'.join(chunks)
        # texts.append(text)
        # labels.append(label)
        # break
            # print("################################################")
            # print(text)
            # print(label)
            # print("################################################")
            # break
        # token_num = text.split()
        # lengths.append(len(token_num))
# dataset = Dataset.from_dict({"text":texts, "label":labels})
# dataset.save_to_disk('./NetCraft_test')