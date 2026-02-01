# Loading Packages
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
random.seed(42)

### Important Variables ###
dual_encoder_temp = 0.3
epochs = 1

data_dir = "/work/mbouthil/datasets/msmarco_synq_2"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split='train')


### MSMARCO ###
class MSMARCO:
    def __init__(self,
                 queries:dict,
                 passages:dict, 
                 qrels:dict,
                 batch_size:int):

        '''
        Data loader for MS MARCO dataset
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
    
    def find_alternative(self, idx):
        return random.choice(self.valid_idx)

    
    def __getitem__(self, idx):

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
    
    # Tokenization
def collate_fn(batch, tokenizer, max_length=128):
    queries = [x['query'] for x in batch]
    positives = [x['positive'] for x in batch]
    # negatives = []

    # for x in batch:
    #     negatives.extend(x['negatives'])

    q_tok = tokenizer(
        queries, 
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    p_tok = tokenizer(
        positives, # + negatives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {"query": q_tok, "passages":p_tok}


class DualEncoder(nn.Module):
    def __init__(self, query_model_name, passage_model_name):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.passage_encoder = AutoModel.from_pretrained(passage_model_name)

    def encode_query(self, **inputs):
        out = self.query_encoder(**inputs)
        return out.last_hidden_state[:, 0]        # CLS
    
    def encode_passage(self, **inputs):
        out = self.passage_encoder(**inputs)
        return out.last_hidden_state[:, 0]        # CLS 
    

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = "cuda" if torch.cuda.is_available() else "cpu"


model = DualEncoder(
    query_model_name="bert-base-uncased",
    passage_model_name="bert-base-uncased"
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


def contrastive_loss(q_emb:Tensor, p_emb:Tensor, temperature:float=dual_encoder_temp) -> Tensor:

    M = q_emb.shape[0]
    N = p_emb.shape[0]

    q_emb = F.normalize(q_emb, dim=-1)
    p_emb = F.normalize(p_emb, dim=-1)
    
    scores = torch.matmul(q_emb, p_emb.T)/temperature
    labels = torch.arange(M) * int(N/M)
    labels = labels.to(device=scores.device)

    loss = F.cross_entropy(scores, labels)
    return loss


batch_sizes = [16, 32, 64, 128, 256] #, 512]
times = []
model.train()

for size in batch_sizes:

    start = time.time()

    dataset = MSMARCO(queries, corpus, qrels, batch_size=size)
    dataloader = DataLoader(
        dataset, 
        batch_size=size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    train_loss = []
    epoch_loss = 0

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
        end = time.time()
        break

    del dataset

    times.append((end-start)/size)


plt.figure(figsize=(12, 12))
plt.suptitle("Batch Size Training Time")
plt.plot(batch_sizes, times, label="Training Loss", linestyle="-", marker="o")
plt.ylabel("Seconds per Sample")
plt.xlabel("Batch Size")
plt.xticks(batch_sizes, batch_sizes)
plt.legend()
plt.style.use('bmh')
plt.savefig("/work/mbouthil/MMATH-CM-Research-Project/RAG/figures/batch_time.png", dpi=300)