# Training of the Dual Encoders using contrastive loss
import pandas as pd
import numpy as np
import random
import os
import torch
import faiss
import json
import re
import shutil
import random
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM
logging.set_verbosity_error()


##### Pre Ambles #####
### Data Subset ###
N = 100_000

### Parameters ###
batch_size = 128
dual_encoder_temp = 0.3
learning_rate = 2e-5
epochs = 1
save_name = "_sq2"


### Loading Dataset ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating subset
print('Creating subset')
keys = set(random.sample([key for key in queries.keys()], N))
queries = dict([(id, query) for id, query in queries.items() if id in keys])
qrels = dict([(qid, pid) for qid, pid in qrels.items() if qid in keys])
print('Subset created')

### MSMARCO class ###
class MSMARCO:
    def __init__(self,
                 queries:dict,
                 passages:dict, 
                 qrels:dict,
                 batch_size:int):

        '''
        Dataset formatter for MS MARCO dataset
        '''

        self.queries = queries
        self.passages = passages
        self.qrels = qrels
        self.qids = list(self.qrels.keys())

        self.batch_size = batch_size
        self.sample_counter = 0
        self.valid_idx = list(range(len(qrels)))
        self.available_pids = set(self.passages.keys())
    
    def __len__(self):
        return len(self.qrels)
    
    def find_alternative(self):
        return random.choice(self.valid_idx)
    
    def __getitem__(self, idx):

        '''
        Given an index, the function returns a valid positive query passage pair (q,p).
        This function is constructed such that a query will only have one corresponding 
        positive passage within its batch.

        Note: If the provided index is invalid, (as another in batch positive is present)
        the function will return an alternative query passage pair

            idx: query idx 
        '''

        sample_selected = False
        idx = idx

        while not sample_selected:

            qid = self.qids[idx]            # Select query id using idx
            query = self.queries[qid]       # Select corresponding query text

            pos_pids = [k for k, v in qrels[qid].items() if v > 0 and k in self.available_pids]    # Selects random pos_id

            if len(pos_pids) < 1:
                self.valid_idx.remove(idx)
                idx = random.choice(self.valid_idx)
                continue

            pos_passage = self.passages[random.choice(pos_pids)]['text']
            self.available_pids = self.available_pids - set(pos_pids)

            self.valid_idx.remove(idx)
            self.sample_counter += 1
            sample_selected = True

        if self.sample_counter == self.batch_size:
            self.sample_counter = 0
            self.available_pids = set(self.passages.keys())
            self.valid_idx = list(range(len(qrels)))
        
        return {"query": query, "positive": pos_passage}
    

dataset = MSMARCO(queries, corpus, qrels, batch_size)


# Tokenization function for batches
def collate_fn(batch, tokenizer, max_length=128):
    queries = [x['query'] for x in batch]
    positives = [x['positive'] for x in batch]

    q_tok = tokenizer(
        queries, 
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    p_tok = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {"query": q_tok, "passages":p_tok}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class DualEncoder(nn.Module):
    def __init__(self, query_model_name, passage_model_name):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.passage_encoder = AutoModel.from_pretrained(passage_model_name)

    def encode_query(self, **inputs):
        out = self.query_encoder(**inputs)
        return out.last_hidden_state[:, 0]
    
    def encode_passage(self, **inputs):
        out = self.passage_encoder(**inputs)
        return out.last_hidden_state[:, 0]
    

# DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda x: collate_fn(x, tokenizer)
)

model = DualEncoder(
    query_model_name="bert-base-uncased",
    passage_model_name="bert-base-uncased"
).to(device)


def contrastive_loss(q_emb:Tensor, p_emb:Tensor, temperature:float=dual_encoder_temp) -> Tensor:

    '''
    This function calculates the constractive loss (Info NCE) of the query 
    embedding matrix and passsage embedding matrix. 

        q_emb: M x N matrix
        p_emb: L x N matrix
        temperature: float (0, 1]
    '''

    M = q_emb.shape[0]                                      # Number of query rows
    L = p_emb.shape[0]                                      # Number of passage rows 

    q_emb = F.normalize(q_emb, dim=-1)                      # Normalizing
    p_emb = F.normalize(p_emb, dim=-1)                      # Normalizing
    
    scores = torch.matmul(q_emb, p_emb.T)/temperature       # Calculating similarity scores (cosine similarity)
    labels = torch.arange(M) * int(L/M)                     # Gathering labels
    labels = labels.to(device=scores.device)

    loss = F.cross_entropy(scores, labels)                  # Calculating Cross Entropy Loss = Info NCE
    return loss


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loss = []


print('\nTraining...\n')
for i in range(epochs):

    model.train()
    epoch_loss = 0
    start = time.time()

    for step, batch in enumerate(dataloader):

        q_inputs = {k: v.to(device) for k, v in batch["query"].items()}
        p_inputs = {k: v.to(device) for k, v in batch["passages"].items()}

        q_emb = model.encode_query(**q_inputs)
        p_emb = model.encode_passage(**p_inputs)

        loss = contrastive_loss(q_emb, p_emb)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del q_emb, p_emb, loss  # Explicitly free memory
        torch.cuda.empty_cache()

    avg_loss = epoch_loss/len(dataloader)
    train_loss.append(avg_loss)
    print(f'Epoch {i+1} Complete')
    print((time.time() - start)/60)

print('Training Complete')

# plotting Loss
plt.figure(figsize=(12, 12))
plt.suptitle("Bi-Encoder Training Loss")
plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss", linestyle="-", marker="o")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.style.use('bmh')
plt.savefig("/work/mbouthil/MMATH-CM-Research-Project/RAG/figures/loss_curve" + save_name + ".png", dpi=300)


# Saving Encoder Weights
save_dir = "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights"
model.query_encoder.save_pretrained(f"{save_dir}/query_encoder" + save_name)
model.passage_encoder.save_pretrained(f"{save_dir}/passage_encoder" + save_name)
tokenizer.save_pretrained(save_dir)

print("Models saved")