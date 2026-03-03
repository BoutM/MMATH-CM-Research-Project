# Training of the Dual Encoders using contrastive loss
import pandas as pd
import numpy as np
import random
import os
import gc
import torch
import faiss
import json
import random
import torch
import time
import sys
import gc
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from packages.marco_dataloader import MSMARCO
from torch.cuda.amp import autocast, GradScaler
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM
# logging.set_verbosity_error()


##########
### Parameters ###
batch_size=256
tau=0.03
learning_rate=2e-5
steps=10_000
plot_loss=True
K=100_000

### Dataset ###
data_dir = "/work/mbouthil/datasets/msmarco_neg_p_100k"

### Model Name ###
model_name = "base_reduced_100k_synq"
##########


passages, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
device = "cuda" if torch.cuda.is_available() else "cpu"
queries = dict(list(queries.items())[:K])
data = MSMARCO(queries, 
               qrels, 
               passages, 
               batch_negatives=1)


### Tokenizer ###
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


### Dual Encoder ###
class DualEncoder(nn.Module):
    def __init__(self, query_model, passage_model):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model)
        self.passage_encoder = AutoModel.from_pretrained(passage_model)

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
    query_model="bert-base-uncased",
    passage_model="bert-base-uncased"
).to(device)


### Loss Function ###
def contrastive_loss(q_emb:Tensor, p_emb:Tensor, tau:float=tau) -> Tensor:

    '''
    This function calculates the constractive loss (Info NCE) of the query 
    embedding matrix and passsage embedding matrix. 

        q_emb: M x N matrix
        p_emb: L x N matrix
        tau: float (0, 1]
    '''

    M = q_emb.shape[0]                                      # Number of query rows
    L = p_emb.shape[0]                                      # Number of passage rows 

    q_emb = F.normalize(q_emb, dim=-1)                      # Normalizing
    p_emb = F.normalize(p_emb, dim=-1)                      # Normalizing
    
    scores = torch.matmul(q_emb, p_emb.T)/tau       # Calculating similarity scores (cosine similarity)
    labels = torch.arange(M) * int(L/M)                     # Gathering labels
    labels = labels.to(device=scores.device)

    loss = F.cross_entropy(scores, labels)                  # Calculating Cross Entropy Loss = Info NCE
    return loss


### Training Loop ###
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=learning_rate)
scaler = GradScaler()
step_loss = []
model.train()

print(model_name, "training in progress...")
for step in tqdm(range(steps), file=sys.stdout):

    start=time.time()
    batch=data.fetch()
    queries, passages = zip(*batch)
    passages=[x for sublist in passages for x in sublist]

    q_inputs=query_tok(queries).to(device)
    p_inputs=passage_tok(passages).to(device)

    with autocast():  # fp16 training
        q_emb=model.encode_query(**q_inputs)
        p_emb=model.encode_passage(**p_inputs)
        loss=contrastive_loss(q_emb, p_emb)

        step_loss.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

print(step_loss[-1])

print('Training complete')