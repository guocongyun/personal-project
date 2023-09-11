import mailbox
import os
import tensorflow as tf 
from tqdm import tqdm
import email
import re
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from datasets import Dataset
from email import policy
texts = []
labels = []
def extract_email_content(mbox_path):
    # Open the mbox file
    c= 0
    mbox = mailbox.mbox(mbox_path)

    # Iterate through all messages in the mbox file
    for i, message in tqdm(enumerate(mbox)):
        # Extract the email content
        body = message.get_payload()
        for b in body:
            try:
                if type(b) != str: b = b.as_string()
            except:
                c += 1
                continue
            # .as_string(policy=policy.SMTPUTF8)
            soup = BeautifulSoup(b, "lxml")
            for script in soup(["class", "id", "name","script", "style"]):
                script.extract()  
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            if chunks and chunks[0] == 'This is a multi-part message in MIME format.':
                chunks = chunks[1:]
            counts = Counter(chunks)
            chunks = [item for item, count in counts.items() if count == 1 and item[:13] != "Content-Type:" ]
            k = len(chunks)
            if k <= 5: continue
            text = '\n'.join(chunks)
            texts.append(text)
            labels.append(1)

    print(c)
        # # Create an output file for each email
        # output_path = os.path.join(output_dir, f'email_{i}.txt')

        # # Save the email content to the output file
        # with open(output_path, 'w', encoding='utf-8') as output_file:
        #     output_file.write(email_content)

# Example usage:
# Specify the path to the mbox file and the directory where you want to save the emails
mbox_path = './Phishingcorpus/phishing0.mbox'
# output_dir = 

# Call the function
extract_email_content(mbox_path)
dataset = Dataset.from_dict({"text":texts, "label":labels})
dataset.save_to_disk('./PhishingCorpus')