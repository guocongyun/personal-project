from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm
from gensim.models import Word2Vec
import gensim.downloader
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import KeyedVectors
import nltk
# nltk.data.path.append(custom_path)
model = KeyedVectors.load_word2vec_format("./word2vec-google-news-300.gz", binary=True)

def tokenize_function(examples):
    tokens = word_tokenize(examples["text"].lower())
    words = [word for word in tokens if word in model.key_to_index]
    if words: examples["input_vec"] = np.mean([model[word] for word in words], axis=0)
    else: examples["input_vec"] =  np.zeros(model.vector_size)
    return examples

def tokens_to_vector(tokens, model):
    words = [word for word in tokens if word in model.key_to_index]
    if words:
        return np.mean([model[word] for word in words], axis=0)
    else:
        return np.zeros(model.vector_size)

def trainer(epochs=20,
            batch_size=1,
            data_path="./data/NetCraft",
            save_path="./new_full",
            pretrained_path=None,
            # ... (other hyperparameters)
            ):
    
    # ... (same code for loading the data)
    dataset = load_from_disk(data_path)
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Define and train the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)


    # Fit the Random Forest classifier
    model.fit(tokenized_dataset["train"]["input_vec"], tokenized_dataset["train"]["label"])

    # Predict on training and validation set
    train_preds = model.predict(tokenized_dataset["train"]["input_vec"])
    valid_preds = model.predict(tokenized_dataset["valid"]["input_vec"])

    # Calculate F1 Score
    train_f1 = f1_score(tokenized_dataset["train"]["label"], train_preds)
    valid_f1 = f1_score(tokenized_dataset["valid"]["label"], valid_preds)

    print('Train F1 Score:', train_f1)
    print('Validation F1 Score:', valid_f1)

    # ... (other logging and saving code)

if __name__ == "__main__":
    trainer()
    