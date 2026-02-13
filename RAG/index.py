# Vector Database Creation
import pandas as pd
import gc
import torch
import json
from transformers import AutoTokenizer, logging, AutoModel
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


### Paths and Directories ###
corpus_path = "/work/mbouthil/datasets/msmarco/corpus.jsonl"
model_name = "base_final"
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