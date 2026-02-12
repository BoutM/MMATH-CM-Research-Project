# Vector Database Creation
import pandas as pd
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

# Loading Data
data_path = "/work/mbouthil/datasets/msmarco/corpus.jsonl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Passage Encoder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
passage_encoder = AutoModel.from_pretrained(
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights/passage_encoder_v1"
).to(device)
passage_encoder.eval()

DIM = 768
NLIST = 4096
M = 64
NBITS = 8
TRAIN_SIZE = 100_000
BATCH_SIZE = 64
CHUNK_SIZE = 5000

quantizer = faiss.IndexFlatIP(DIM)
index = faiss.IndexIVFPQ(
    quantizer,
    DIM,
    NLIST,
    M,
    NBITS
)
index.nprobe = 16

train_buf = []
train_count = 0
is_trained = False

meta_file = open(
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/retrieval_data/passage_metadata_v1.jsonl",
    "w"
)

global_idx = 0

for chunk in stream_msmarco_chunks(data_path, CHUNK_SIZE):
    for i in range(0, len(chunk), BATCH_SIZE):
        batch = chunk[i : i + BATCH_SIZE]
        passages = [x[1] for x in batch]
        doc_ids = [x[0] for x in batch]
        
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
            faiss.normalize_L2(emb)
        
        if not is_trained:
            train_buf.append(emb)
            # Write metadata for training vectors
            for doc_id in doc_ids:
                meta_file.write(json.dumps({
                    "idx": global_idx,
                    "doc_id": doc_id
                }) + "\n")
                global_idx += 1
            
            train_count += emb.shape[0]
            if train_count >= TRAIN_SIZE:
                print(f"Training FAISS index on {train_count} vectors")
                train_vecs = np.vstack(train_buf)
                index.train(train_vecs)
                index.add(train_vecs)
                is_trained = True
                train_buf = []
                del train_vecs
                print("✓ FAISS index trained")
        elif is_trained:
            index.add(emb)
            for doc_id in doc_ids:
                meta_file.write(json.dumps({
                    "idx": global_idx,
                    "doc_id": doc_id
                }) + "\n")
                global_idx += 1
        
        del emb, inputs
        torch.cuda.empty_cache()

meta_file.close()
faiss.write_index(
    index,
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/retrieval_data/passage_v1.index"
)
print(f"✓ Index creation complete. Total vectors indexed: {global_idx}")