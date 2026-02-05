# Vector Database Creation
import pandas as pd
import gc
import torch
from transformers import AutoTokenizer, logging, AutoModel
logging.set_verbosity_error()
import numpy as np
import os
from torch import Tensor
from torch.utils.data import DataLoader
import faiss
import json
from beir.datasets.data_loader import GenericDataLoader


### Data Streamer...
# Streaming Code
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


### Loading data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, _, _  = GenericDataLoader(data_folder=data_dir).load(split="train")
N = len(corpus)
del corpus

data_path = data_dir + "/corpus.jsonl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Loading Passage Encoder ###
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
passage_encoder = AutoModel.from_pretrained(
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights/passage_encoder_v1"
).to(device)
passage_encoder.eval()

### Formating Directories ###
output_dir = "/work/mbouthil/MMATH-CM-Research-Project/RAG/retrieval_data"
meta_file = open(f"{output_dir}/passage_metadata_v1.jsonl", "w")
embeddings_path = f"{output_dir}/embeddings_temp_v1.npy"



d = 768
total_estimated = N

embedding_memmap = np.memmap(
    embeddings_path,
    dtype='float32',
    mode='w+',
    shape=(total_estimated, d)
)


batch_size = 64
chunksize = 5000
total_passages = 0


# Writting Embeddings:
for chunk in stream_msmarco_chunks(data_path, chunksize):

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
            embedding_memmap[total_passages:total_passages + batch_size] = emb
            total_passages += batch_size

            del inputs, emb

        if i % (batch_size*10) == 0:
            torch.cuda.empty_cache()

    del chunk
    gc.collect()

embedding_memmap.flush()
del embedding_memmap


### Writting index ###
embeddings = np.memmap(
    embeddings_path,
    dtype='float32',
    mode='r',
    shape=(N, 768)
)

index = faiss.IndexFlatIP(d)
chunk_size = 100_000

for start_idx in range(0, total_passages, chunk_size):
    end_idx = min(start_idx + chunk_size, total_passages)
    chunk_emb = np.array(embeddings[start_idx:end_idx])
    index.add(chunk_emb)
    del chunk_emb
    gc.collect()

faiss.write_index(index, f"{output_dir}/passage_v1.index")