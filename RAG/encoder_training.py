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

syn_pas_debug = True

### Parameters ###
batch_size=256
dual_encoder_temp=0.3
learning_rate=1e-4
epochs=30
save_name= "_syn_pas_1"


### Loading Dataset ###
data_dir = "/work/mbouthil/datasets/msmarco_syn_pas_1"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
device = "cuda" if torch.cuda.is_available() else "cpu"

if syn_pas_debug == True:
    for key, value in corpus.items():
        corpus[key] = value['text']


class MSMARCO:
    def __init__(self, queries, passages, qrels):

        self.queries = queries
        self.passages = passages
        self.qrels = qrels

    def __len__(self):
        return (len(self.queries))
    
    def __getitem__(self, pair):

        qid, pid = pair

        query = self.queries[qid]
        passage = self.passages[pid]['text']

        return {"query": query, "positive": passage}


class BatchSampler:
    def __init__(self, 
                queries:dict, 
                passages:dict, 
                qrels:dict, 
                batch_size:int=batch_size, 
                shuffle:bool=True):

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.qrels = qrels
        self.pids = set(passages.keys())
        self.qids = list(queries.keys())

        self.batches = self._create_batches(batch_size)

    def _create_batches(self, batch_size):

        batches = []
        remaining_qids = self.qids.copy()

        while len(remaining_qids) >= self.batch_size:

            random.shuffle(remaining_qids)
            current_batch = []
            unavailable_pids = set()
            qids_to_remove = []

            for qid in remaining_qids:
                pos_pids = [k for k, v in self.qrels[qid].items() 
                            if k not in unavailable_pids and v > 0]

                if pos_pids:
                    pid = random.choice(pos_pids)
                    current_batch.append((qid, pid))
                    unavailable_pids.update(pos_pids)
                    qids_to_remove.append(qid)

                    if len(current_batch) == self.batch_size:
                        break
                else:
                    continue

            for qid in qids_to_remove:
                remaining_qids.remove(qid)
            batches.append(current_batch)

        return batches
    
    
    def __iter__(self):
        '''
        Iterator that yeilds batches. Called at the beginning of the epoch
        '''
        if self.shuffle:
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            for idx in indices:
                yield self.batches[idx]
        else:
            for batch in self.batches:
                yield batch

    def __len__(self):
        return len(self.batches)


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

dataset = MSMARCO(queries, corpus, qrels)
batch_sampler = BatchSampler(
    queries=queries,
    passages=corpus,
    qrels=qrels)

dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda x: collate_fn(x, tokenizer),
    persistent_workers=True,
    prefetch_factor=2
)


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
scaler = GradScaler()
train_loss = []

print(f'\n{save_name} training in progress...\n')
for i in range(epochs):

    model.train()
    epoch_loss = 0
    start = time.time()

    for step, batch in enumerate(dataloader):

        q_inputs = {k: v.to(device) for k, v in batch["query"].items()}
        p_inputs = {k: v.to(device) for k, v in batch["passages"].items()}

        with autocast():  # fp16 training
            q_emb = model.encode_query(**q_inputs)
            p_emb = model.encode_passage(**p_inputs)
            loss = contrastive_loss(q_emb, p_emb)

        epoch_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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