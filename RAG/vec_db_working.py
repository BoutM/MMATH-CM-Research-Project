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


# Loading Data
data_path = "/work/mbouthil/datasets/msmarco/corpus.jsonl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Passage Encoder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
passage_encoder = AutoModel.from_pretrained(
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights/passage_encoder_v1"
).to(device)
passage_encoder.eval()

BATCH_SIZE = 64
CHUNK_SIZE = 5000

meta_file = open(
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/retrieval_data/passage_metadata_test.jsonl",
    "w"
)

d = 768  # or whatever your embedding dimension is
index = faiss.IndexFlatIP(d)  # That's it!

# Your existing code works as-is:
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
            faiss.normalize_L2(emb)  # Keep this one
            index.add(emb)           # Add immediately after normalizing


meta_file.close()
faiss.write_index(
    index,
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/retrieval_data/passage_test.index"
)

