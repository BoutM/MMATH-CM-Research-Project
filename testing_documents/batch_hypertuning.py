# Training of the Dual Encoders using contrastive loss
import numpy as np
import random
import torch
import random
import torch
import time
import sys
import gc
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoModel, AutoTokenizer

##########
### Parameters ###
batch_size=[32, 64, 128, 256, 512]
tau=0.03
lr=2e-5
steps=10_000
data_dir = "/work/mbouthil/datasets/msmarco"            # Using base dataset
##########


### Loading data ###
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
device = "cuda" if torch.cuda.is_available() else "cpu"


### Dataset ###
class MSMARCO:
    def __init__(self, 
                queries:dict, 
                passages:dict, 
                qrels:dict, 
                batch_size:int=256, 
                shuffle:bool=True):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.qrels = qrels
        self.pids = set(passages.keys())
        self.qids = list(queries.keys())
        self.passages = passages
        self.queries = queries

    def fetch_batch(self):

        batch = []
        remaining_qids = self.qids.copy()
        unavailable_pids = set()

        while len(batch) < self.batch_size:

            qid = remaining_qids.pop(random.randrange(len(remaining_qids)))
            pos_pids = [k for k, v in self.qrels[qid].items() 
                    if k not in unavailable_pids and v > 0]

            if pos_pids:
                pid = random.choice(pos_pids)
                batch.append((self.queries[qid], self.passages[pid]['text']))
            unavailable_pids.update(pos_pids)

        return batch

### Tokenizer ###
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

def passage_tok(passages:list[str], max_length:int=128) -> dict:
    with torch.no_grad():
        inputs = tokenizer(
            passages, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(device)
    return inputs


### Dual Encoder ###
class DualEncoder(nn.Module):
    def __init__(self, query_model, passage_model):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model)
        self.passage_encoder = AutoModel.from_pretrained(passage_model)

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

    def encode_query(self, **inputs):
        out = self.query_encoder(**inputs)
        return self.mean_pool(out.last_hidden_state, inputs["attention_mask"])

    def encode_passage(self, **inputs):
        out = self.passage_encoder(**inputs)
        return self.mean_pool(out.last_hidden_state, inputs["attention_mask"])
    
model = DualEncoder(
    query_model="bert-base-uncased",
    passage_model="bert-base-uncased"
).to(device)


### Contrastive Loss Function ###
def contrastive_loss(q_emb:Tensor, p_emb:Tensor, tau:float=0.03) -> Tensor:

    '''
    This function calculates the constractive loss (Info NCE) of the query 
    embedding matrix and passsage embedding matrix. 

        q_emb: M x N matrix
        p_emb: L x N matrix
        tau: float (0, 1]
    '''

    M = q_emb.shape[0]                                      # Number of query rows
    L = p_emb.shape[0]                                      # Number of passage rows 

    q_emb = F.normalize(q_emb, dim=-1)                      # Normalizing
    p_emb = F.normalize(p_emb, dim=-1)                      # Normalizing
    
    scores = torch.matmul(q_emb, p_emb.T)/tau       # Calculating similarity scores (cosine similarity)
    labels = torch.arange(M) * int(L/M)                     # Gathering labels
    labels = labels.to(device=scores.device)

    loss = F.cross_entropy(scores, labels)                  # Calculating Cross Entropy Loss = Info NCE
    return loss


### Training using various tau values ###
model_losses = []
avg_loss = []
i = 0

for size in batch_size:
    Dataset = MSMARCO(queries, corpus, qrels, batch_size=size)
    model = DualEncoder(
        query_model="bert-base-uncased",
        passage_model="bert-base-uncased"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()
    step_loss = []
    model.train()

    print(f"/nbatch_size={size} model training in progress...")
    for step in tqdm(range(steps), file=sys.stdout):

        start = time.time()
        batch = Dataset.fetch_batch()
        q_data, p_data = zip(*batch)

        q_inputs = query_tok(q_data).to(device)
        p_inputs = passage_tok(p_data).to(device)

        with autocast():  # fp16 training
            q_emb = model.encode_query(**q_inputs)
            p_emb = model.encode_passage(**p_inputs)
            loss = contrastive_loss(q_emb, p_emb, tau=tau)

            step_loss.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model_losses.append(step_loss)
    last_5000_loss = step_loss[-5000:]
    avg_loss.append(np.mean(last_5000_loss))
    i+=1
    print(f"Model {i} training complete")


### Plotting Model Losses ###
fig, axes = plt.subplots(1, len(batch_size), figsize=(24, 12))
x = range(1, steps+1)

for i in range(len(batch_size)):
    axes[i].plot(x, model_losses[i])
    axes[i].set_title(f"Batch_size={batch_size[i]}")
    
    axes[i].text(
        0.98, 0.98,
        f"Avg last 5000: {avg_loss[i]:.4f}",
        transform=axes[i].transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

fig.supxlabel("Steps")
fig.supylabel("Loss")

plt.tight_layout()
plt.style.use('bmh')
plt.savefig(f"/work/mbouthil/MMATH-CM-Research-Project/figures/Batch_hypertune.png", dpi=300)



    