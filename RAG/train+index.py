# Training of the Dual Encoders using contrastive loss
import pandas as pd
import numpy as np
import random
import os
import gc
import torch
import faiss
import json
import re
import shutil
import random
import torch
import time
import sys
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
# logging.set_verbosity_error()


### Parameters ###
batch_size=256
dual_encoder_temp=0.3
learning_rate=1e-4
steps=200_000
plot_loss=True

### Dataset ###
data_dir = "/work/mbouthil/datasets/msmarco_syn_p"


### Model Name ###
model_name = "syn_p"


corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
device = "cuda" if torch.cuda.is_available() else "cpu"


### Dataset ###
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


### Training Loop ###
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()
step_loss = []
model.train()

print(model_name, "training in progress...")
for step in tqdm(range(steps), file=sys.stdout):

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


### Plotting Loss ###
if plot_loss==True:
    plt.figure(figsize=(12, 12))
    plt.suptitle("Bi-Encoder Training Loss")
    plt.plot(range(1, len(step_loss)+1), step_loss, label="Training Loss", linestyle="-")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.legend()
    plt.style.use('bmh')
    plt.savefig(f"/work/mbouthil/MMATH-CM-Research-Project/RAG/figures/loss_curve_{model_name}.png", dpi=300)


### Saving Encoder Weights ###
save_dir = "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights"
model.query_encoder.save_pretrained(f"{save_dir}/query_encoder_{model_name}")
model.passage_encoder.save_pretrained(f"{save_dir}/passage_encoder_{model_name}")


### Cleaning Env ###
del queries, corpus, qrels

### Paths and Directories ###
corpus_path = "/work/mbouthil/datasets/msmarco/corpus.jsonl"
output_dir = "/work/mbouthil/MMATH-CM-Research-Project/RAG/retrieval_data"
embeddings_path = f"{output_dir}/embeddings_{model_name}.npy"
device = "cuda" if torch.cuda.is_available() else "cpu"



### Creating Vector Database ###
def stream_msmarco_chunks(path, chunk_size=5000):
    buffer = []
    with open(path, "r") as f:
        for line in f:
            doc = json.loads(line)
            buffer.append((doc["_id"], doc["text"]))
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


### Loading Passage Encoder ###
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
passage_encoder = AutoModel.from_pretrained(
    f"/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights/passage_encoder_{model_name}"
    ).to(device)
passage_encoder.eval()


### variables ###
N = 8_841_823                       # Amounts of passages in the original data corpus
d = 768                             # Embedding size


### Creating memmap ###
embedding_memmap = np.memmap(
    embeddings_path,
    dtype='float32',
    mode='w+',
    shape=(N, d)
)
batch_size = 64
chunksize=5000
N = 0



### Writting Embeddings ###
print('Writing index')
pbar=tqdm(total=np.ceil(N/chunksize), file=sys.stdout)
for chunk in stream_msmarco_chunks(corpus_path, chunksize):

    for i in range(0, len(chunk), batch_size):
        batch = chunk[i : i + batch_size]
        passages = [x[1] for x in batch]
        
        with torch.no_grad():
            inputs = tokenizer(
                passages,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            emb = passage_encoder(**inputs).last_hidden_state[:, 0]
            emb = emb.float().cpu().numpy()
            emb = np.ascontiguousarray(emb, dtype=np.float32)

            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / norms
            # faiss.normalize_L2(emb)  # Keep this one
            # index.add(emb)           # Add immediately after normalizing

            batch_size = emb.shape[0]
            embedding_memmap[N:N + batch_size] = emb
            N += batch_size

            del inputs, emb

        if i % (batch_size*10) == 0:
            torch.cuda.empty_cache()

    del chunk
    pbar.update(1)
    gc.collect()
pbar.close

embedding_memmap.flush()
del embedding_memmap


### Reading memmap ###
embeddings = np.memmap(
    embeddings_path,
    dtype='float32',
    mode='r',
    shape=(N, d)
)

### Writting index ###
index = faiss.IndexFlatIP(d)
chunk_size = 100_000

for start_idx in range(0, N, chunk_size):
    end_idx = min(start_idx + chunk_size, N)
    chunk_emb = np.array(embeddings[start_idx:end_idx])
    index.add(chunk_emb)
    del chunk_emb
    gc.collect()
    break

faiss.write_index(index, f"{output_dir}/passage_{model_name}.index")

print("Traing and vector database construction completed")