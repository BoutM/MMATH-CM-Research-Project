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
_, train_queries, _ = GenericDataLoader(data_folder=data_dir).load(split='train')
_, test_queries, _ = GenericDataLoader(data_folder=data_dir).load(split='dev')

train_q = [(q_id, query) for q_id, query in train_queries.items()]
test_q = [(q_id, query) for q_id, query in test_queries.items()]


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
    "bert-base-uncased"
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

train_batches = list(batch_splits(train_q))
test_batches = list(batch_splits(test_q))


sim_info = []

for train_batch in train_batches:

    train_ids, train_batch = zip(*train_batch)
    train_emb = encode_query(train_batch)
    
    for test_batch in test_batches:

        test_ids, test_batch = zip(*test_batch)
        test_emb = encode_query(test_batch)

        sim = torch.matmul(test_emb, train_emb.T).detach().cpu().numpy()
        indices = np.argwhere(sim >= 0.75)

        sim_info.extend([(test_ids[i], train_ids[j], sim[i,j], test_queries[str(test_ids[i])], train_queries[str(train_ids[j])]) 
                         for i, j in indices])
        break
    break


dir = "/work/mbouthil/datasets/test_results"
path = os.path.join(dir, "sim_info_test.jsonl")


with open(path, "w") as f:
    for test_id, train_id, sim_score, test_text, train_text in sim_info:
        entry = {"train_id": train_id, "test_id": test_id, "sim_score": float(sim_score), "train_query": train_text, "test_query": test_text}
        f.write(json.dumps(entry) + "\n")