import pandas as pd
import torch
import os
import gc
os.environ["HF_HUB_DISABLE_PROGREvSS_BARS"] = "1"
from huggingface_hub import login
from transformers import AutoTokenizer, logging, AutoModel
logging.set_verbosity_error()
import numpy as np
from dotenv import load_dotenv
from torch import Tensor
import faiss 
from beir.datasets.data_loader import GenericDataLoader
import torch.nn.functional as F
from beir.retrieval.evaluation import EvaluateRetrieval


'''
This script evaluates the performance of the specified model using nDCG@10, MMR@10, Recall@100, and Recall@1000.
'''

model_name = "r400s_sq-v2"
dir= "/work/mbouthil/MMATH-CM-Research-Project/retreiver_training/retrieval_data/"


# try:
index = faiss.read_index(f"{dir}{model_name}.index")
print('Index available')
# except:
#     print('Index unavailable: reading embeddings')
#     embeddings_path = f"{dir}embeddings_{model_name}.npy"
#     N = 8_841_823                     
#     d = 768  

#     ### Reading memmap ###
#     embeddings = np.memmap(
#         embeddings_path,
#         dtype='float32',
#         mode='r',
#         shape=(N, d)
#     )

#     print("Writting embeddings")
#     ### Writting index ###
#     index = faiss.IndexFlatIP(d)
#     chunk_size = 100_000

#     for start_idx in range(0, N, chunk_size):
#         end_idx = min(start_idx + chunk_size, N)
#         chunk_emb = embeddings[start_idx:end_idx]
#         index.add(chunk_emb)
#         del chunk_emb
#         gc.collect()

#     faiss.write_index(index, f"{dir}{model_name}.index")
#     del embeddings, index
#     gc.collect()

#     # Loading Index
#     index = faiss.read_index(f"{dir}{model_name}.index")


print('Evaluation in progress...')
### Loading Eval Data ###
_, dev_queries, dev_qrels = GenericDataLoader(data_folder="/work/mbouthil/datasets/msmarco").load(split="dev")
dev_info = list(dev_queries.items())

# Loading Query Encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
query_encoder = AutoModel.from_pretrained(
    f"/work/mbouthil/MMATH-CM-Research-Project/retreiver_training/model_weights/query_encoder_{model_name}"
).to(device)
query_encoder.eval()

def encode_query(query:str) -> Tensor:

    queries = [query] if isinstance(query, str) else query

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

    return emb.cpu()


def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]

batches = batch_splits(dev_info)


results = {}
for batch in batches:

    q_ids, queries = zip(*batch)
    N = range(len(q_ids))
    
    q_emb = encode_query(queries).numpy()
    scores, pids = index.search(q_emb, 1000)
    
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
print(f"NDCG@10: {ndcg['NDCG@10']}\nMRR@10: {mrr['MRR@10']}\nRecall@100: {recall['Recall@100']}\nRecall@1000: {recall['Recall@1000']}")
print("\n")

scores = {
    "NDCG@10": ndcg['NDCG@10'],
    "MRR@10": mrr['MRR@10'],
    "Recall@100": recall['Recall@100'],
    "Recall@1000": recall['Recall@1000']}

scores = pd.DataFrame(scores, index=[0])
scores.to_csv(f"/work/mbouthil/MMATH-CM-Research-Project/results/{model_name}_results.csv", index=False)

print("Evaluation Complete")