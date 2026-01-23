# Vector Database Creation

import pandas as pd
import torch
from transformers import AutoTokenizer, logging, AutoModel
logging.set_verbosity_error()
import numpy as np
import os
from torch import Tensor
import faiss
import json
from beir.datasets.data_loader import GenericDataLoader

# Loading Data
data_dir = "/work/mbouthil/projects/research_project/RAG/datasets/msmarco"
corpus, _, _ = GenericDataLoader(data_folder=data_dir).load(split="train")
passages = [corpus[doc_id]["text"] for doc_id in corpus][:10]


# Selecting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading Passage Encoder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
passage_encoder = AutoModel.from_pretrained(
    "/work/mbouthil/projects/research_project/RAG/model_weights/passage_encoder"
).to(device)
passage_encoder.eval()


# Encoding 
def encode_passage(notes:list[str], batch_size:int=32) -> Tensor:

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(notes), batch_size):
            batch = notes[i:i+batch_size]

            inputs = tokenizer(
                batch, 
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)

            outputs = passage_encoder(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0] 

            embeddings.append(cls_embeddings.cpu())

    return torch.cat(embeddings, 0)

# Creating embeddings
embeddings = encode_passage(passages)

# Moving to CPU
embeddings_np = embeddings.cpu().numpy().astype("float32")
np.save('/work/mbouthil/projects/research_project/RAG/retrieval_data/passage_embeddings.npy', embeddings_np)

# Creating FAISS Index
N, d = embeddings_np.shape
index = faiss.IndexFlatIP(d)
index.add(embeddings)
faiss.write_index(index, "/work/mbouthil/projects/research_project/RAG/retrieval_data/passage.index")

# Creating FAISS metadata.json
with open("/work/mbouthil/projects/research_project/RAG/retrieval_data/passage_metadata.jsonl", "w") as f:
    for idx, passage in enumerate(passages):
        record = {
            "id": idx,
            "text": passage
        }
        f.write(json.dumps(record) + "\n")

