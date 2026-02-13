# Importing packages
import pandas as pd
import torch
import os
import math
os.environ["HF_HUB_DISABLE_PROGREvSS_BARS"] = "1"
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, AutoModel
logging.set_verbosity_error()
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
from torch import Tensor
import faiss 
import json
from beir.datasets.data_loader import GenericDataLoader
import torch.nn.functional as F
from beir.retrieval.evaluation import EvaluateRetrieval

### Model Name ###
model_name = "syn_que_final"


### Loading Data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, dev_queries, dev_qrels = GenericDataLoader(data_folder=data_dir).load(split="dev")
dev_info = [(key, value) for key, value in dev_queries.items()]

# Loading Index
index = faiss.read_index(f"/work/mbouthil/MMATH-CM-Research-Project/RAG/retrieval_data/passage_{model_name}.index")
print(index.ntotal)

# Loading Query Encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
query_encoder = AutoModel.from_pretrained(
    f"/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights/query_encoder_{model_name}"
).to(device)
query_encoder.eval()

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


def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]

batches = batch_splits(dev_info)


results = {}
for batch in batches:

    q_ids, queries = zip(*batch)
    N = range(len(q_ids))
    
    q_emb = encode_query(queries).detach().cpu().numpy()
    scores, pids = index.search(q_emb, 100)
    
    pids = [[str(pid) for pid in pids[i]] for i in N]
    scores = [[float(score) for score in scores[i]] for i in N]
    
    batch_results = {
        q_ids[i]: dict(zip(pids[i], scores[i])) 
        for i in N
    }
    
    results.update(batch_results)


ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
    dev_qrels,
    results, 
    k_values=[1, 3, 5, 10, 100, 1000]
)

mrr = EvaluateRetrieval.evaluate_custom(
    dev_qrels, 
    results, 
    k_values=[10], 
    metric="mrr"
)


print("\n")
print(f"NDCG@10: {ndcg['NDCG@10']}")
print("\n")
print(f"Recall@100: {recall['Recall@100']}")
print("\n")
print(f"MRR@10: {mrr['MRR@10']}") 

