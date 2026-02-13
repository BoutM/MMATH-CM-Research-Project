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
from tqdm import tqdm
from torch import Tensor
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
import torch.nn.functional as F
from beir.datasets.data_loader import GenericDataLoader
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM
logging.set_verbosity_error()


# Loading in datasets
data_dir = "/work/mbouthil/datasets/msmarco"
train_corpus, train_queries, train_qrels = GenericDataLoader(data_folder=data_dir).load(split='train')
test_corpus, test_queries, test_qrels = GenericDataLoader(data_folder=data_dir).load(split='dev')


test_q = [(q_id, query) for q_id, query in test_queries.items()]
train_q = [(q_id, query) for q_id, query in train_queries.items()]


device = "cuda" if torch.cuda.is_available() else "cpu"
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


query_encoder = AutoModel.from_pretrained(
    "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights/query_encoder_v01"
).to(device)


def encode_query(query:str) -> Tensor:

    queries = [query] if isinstance(query, str) else query
    embeddings = []

    with torch.no_grad():
        inputs = tokenizer(
            queries, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=32
        ).to(device)

    # Mean Pooling
    out = query_encoder(**inputs)
    last_hidden_state = out.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    emb = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

    emb = F.normalize(emb, p=2, dim=-1)
    embeddings.append(emb.cpu())

    return torch.cat(embeddings, 0)


def batch_splits(item:list, batch_size:int=100):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]

train_batches = batch_splits(train_q)
test_batches = batch_splits(test_q)


sim_q_ids = []

for train_batch in train_batches:

    train_ids, train_batch = zip(*train_batch)
    train_emb = encode_query(train_batch)
    
    for test_batch in test_batches:

        test_ids, test_batch = zip(*test_batch)
        test_emb = encode_query(test_batch)

        sim = torch.matmul(test_emb, train_emb.T).detach().cpu().numpy()
        indices = np.where(sim >= 0.7)

        q_test_ids = [test_ids[i] for i in set(indices[0])]
        q_train_ids = [train_ids[i] for i in set(indices[1])]
        sim_q_ids.extend(list(zip(q_train_ids, q_test_ids)))


dir = "/work/mbouthil/MMATH-CM-Research-Project/RAG/test_results"
path = os.path.join(dir, "sim_ids.jsonl")

with open(path, "w") as f:
    for train_id, test_id in sim_q_ids:
        entry = {"train_id": train_id, "test_id": test_id}
        f.write(json.dumps(entry) + "\n")