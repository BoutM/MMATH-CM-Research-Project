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
from torch.cuda.amp import autocast, GradScaler
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM
logging.set_verbosity_error()

syn_pas = False

### Parameters ###
batch_size=256
dual_encoder_temp=0.3
learning_rate=2e-5
steps=1
save_name = "_TEST"

### Loading Dataset ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
device = "cuda" if torch.cuda.is_available() else "cpu"

if syn_pas==True:
    for key in corpus.keys():
        corpus[key] = corpus[key]['text']


class MSMARCO:
    def __init__(self, 
                queries:dict, 
                passages:dict, 
                qrels:dict, 
                batch_size:int=256, 
                shuffle:bool=True):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.qrels = qrels
        self.pids = set(passages.keys())
        self.qids = list(queries.keys())
        self.passages = passages
        self.queries = queries

    def fetch_batch(self):

        batch = []
        remaining_qids = self.qids.copy()
        unavailable_pids = set()

        while len(batch) < self.batch_size:

            qid = remaining_qids.pop(random.randrange(len(remaining_qids)))
            pos_pids = [k for k, v in self.qrels[qid].items() 
                    if k not in unavailable_pids and v > 0]

            if pos_pids:
                pid = random.choice(pos_pids)
                batch.append((self.queries[qid], self.passages[pid]['text']))
            unavailable_pids.update(pos_pids)

        return batch
    

Dataset = MSMARCO(queries, corpus, qrels)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def query_tok(queries:list[str], max_length:int=32) -> dict:
    with torch.no_grad():
        inputs = tokenizer(
            queries, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(device)
    return inputs

def passage_tok(passages:list[str], max_length:int=128) -> dict:
    with torch.no_grad():
        inputs = tokenizer(
            passages, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(device)
    return inputs
    

class DualEncoder(nn.Module):
    def __init__(self, query_model_name, passage_model_name):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.passage_encoder = AutoModel.from_pretrained(passage_model_name)

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

    def encode_query(self, **inputs):
        out = self.query_encoder(**inputs)
        return self.mean_pool(out.last_hidden_state, inputs["attention_mask"])

    def encode_passage(self, **inputs):
        out = self.passage_encoder(**inputs)
        return self.mean_pool(out.last_hidden_state, inputs["attention_mask"])

    
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
    print(M)
    print(L)

    q_emb = F.normalize(q_emb, dim=-1)                      # Normalizing
    p_emb = F.normalize(p_emb, dim=-1)                      # Normalizing
    
    scores = torch.matmul(q_emb, p_emb.T)/temperature       # Calculating similarity scores (cosine similarity)
    labels = torch.arange(M) * int(L/M)                     # Gathering labels
    labels = labels.to(device=scores.device)

    loss = F.cross_entropy(scores, labels)                  # Calculating Cross Entropy Loss = Info NCE
    return loss


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()
step_loss = []
model.train()

print(save_name, " training in progress...")
for step in range(steps):

    start = time.time()
    batch = Dataset.fetch_batch()
    queries, passages = zip(*batch)

    q_inputs = query_tok(queries).to(device)
    p_inputs = passage_tok(passages).to(device)

    with autocast():  # fp16 training
        q_emb = model.encode_query(**q_inputs)
        p_emb = model.encode_passage(**p_inputs)
        loss = contrastive_loss(q_emb, p_emb)

        step_loss.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

print('Training complete')

# plotting Loss
plt.figure(figsize=(12, 12))
plt.suptitle("Bi-Encoder Training Loss")
plt.plot(range(1, len(step_loss)+1), step_loss, label="Training Loss", linestyle="-", marker="o")
plt.ylabel("Loss")
plt.xlabel("Step")
plt.legend()
plt.style.use('bmh')
plt.savefig("/work/mbouthil/MMATH-CM-Research-Project/RAG/figures/loss_curve_" + save_name + ".png", dpi=300)

# Saving Encoder Weights
save_dir = "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights"
model.query_encoder.save_pretrained(f"{save_dir}/query_encoder_" + save_name)
model.passage_encoder.save_pretrained(f"{save_dir}/passage_encoder_" + save_name)
tokenizer.save_pretrained(save_dir)

print("Encoder models saved")