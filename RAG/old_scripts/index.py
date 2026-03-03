# Vector Database Creation
import pandas as pd
import gc
import torch
import json
from transformers import AutoTokenizer, logging, AutoModel
import torch.nn.functional as F
logging.set_verbosity_error()
import numpy as np
from tqdm import tqdm
import os
import sys
from torch import Tensor
from torch.utils.data import DataLoader
import faiss
import json
from beir.datasets.data_loader import GenericDataLoader

model_name = "r300s_synp"

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
entries = 8_841_823                       # Amounts of passages in the original data corpus
dim = 768                                 # Embedding size


### Creating memmap ###
embedding_memmap = np.memmap(
    embeddings_path,
    dtype='float32',
    mode='w+',
    shape=(entries, dim)
)
batch_size = 64
chunksize=5000


### Writting Embeddings ###
print('Writing index')
total_batches=np.ceil(entries/chunksize)
counter=0
N = 0

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

            out = passage_encoder(**inputs)
            last_hidden_state = out.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            emb = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            emb = F.normalize(emb, p=2, dim=-1)

            emb = emb.float().cpu().numpy()
            emb = np.ascontiguousarray(emb, dtype=np.float32)

            batch_size = emb.shape[0]
            embedding_memmap[N:N + batch_size] = emb
            N += batch_size

            del inputs, emb

        if i % (batch_size*10) == 0:
            torch.cuda.empty_cache()

    del chunk
    counter += 1
    print(f"Chunk {counter} of {total_batches} completed")

embedding_memmap.flush()
del embedding_memmap


### Reading memmap ###
embeddings = np.memmap(
    embeddings_path,
    dtype='float32',
    mode='r',
    shape=(entries, dim)
)


### Writting index ###
index = faiss.IndexFlatIP(dim)
chunk_size = 100_000

for start_idx in range(0, N, chunk_size):
    end_idx = min(start_idx + chunk_size, N)
    chunk_emb = np.array(embeddings[start_idx:end_idx])
    index.add(chunk_emb)
    del chunk_emb
    gc.collect()

faiss.write_index(index, f"{output_dir}/passage_{model_name}.index")

print("Traing and vector database construction completed")