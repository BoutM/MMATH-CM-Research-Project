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


class BatchSampler:
    def __init__(self, queries, passages, qrels, batch_size, shuffle=True):

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

            print("starting batch")

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
                        print('batch completed')
                        break
                
                else:
                    continue

            for qid in qids_to_remove:
                remaining_qids.remove(qid)

            batches.append(current_batch)

        return batches
    
    def __iter__(self):
        '''
        Iteratior that yeilds batches. Called at the beginning of the epoch
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

    dataset = MSMARCO(queries, corpus, qrels)
    batch_sampler = BatchSampler(
        queries=queries,
        passages=corpus,
        qrels=qrels,
        batch_size=128,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, tokenizer),
        persistent_workers=True,
        prefetch_factor=2
    )

    train_loss = []
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
        end = time.time()
        break

    times.append((end-start)/size)
    del dataset


plt.figure(figsize=(12, 12))
plt.suptitle("Batch Size Training Time")
plt.plot(batch_sizes, times, label="Training Loss", linestyle="-", marker="o")
plt.ylabel("Seconds per Sample")
plt.xlabel("Batch Size")
plt.xticks(batch_sizes, batch_sizes)
plt.legend()
plt.style.use('bmh')
plt.savefig("/work/mbouthil/MMATH-CM-Research-Project/RAG/figures/batch_time.png", dpi=300)