# Training of the Dual Encoders using contrastive loss

import pandas as pd
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


### Data Subset ###
N = 100_000


### Important Variables ###
training_batch_size = 256
dual_encoder_temp = 0.3
save_name = "_sq2"
epochs = 20


# Loading Dataset
data_dir = "/work/mbouthil/datasets/msmarco_synq_2"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")

# queries = dict([(key, query) for key, query in queries.items()][:int(p*len(queries))])
# qrels = dict([(qid, pids) for qid, pids in qrels.items() if qid in queries.keys()])

random.seed(42)
keys = random.sample([key for key in queries.keys()], N)
queries = dict([(id, query) for id, query in queries.items() if id in keys])
qrels = dict([(qid, pid) for qid, pid in qrels.items() if qid in keys])


class MSMARCO:
    def __init__(self,
                 queries:dict,
                 passages:dict, 
                 qrels:dict, 
                 num_negatives:int=number_of_negatives):

        '''
        Data loader for MS MARCO dataset
        '''

        self.queries = queries
        self.passages = passages
        self.qrels = qrels
        self.qids = list(self.qrels.keys())
        self.num_negatives = num_negatives
    
    def __len__(self):
        return len(self.qrels)
    
    def __getitem__(self, idx):
        qid = self.qids[idx]
        query = self.queries[qid]

        pos_pids = [k for k, v in self.qrels[qid].items() if v > 0]
        pos_passage = self.passages[random.choice(pos_pids)]['text']

        # Sample negatives directly from passages dict
        neg_passages = []
        available_pids = list(self.passages.keys())
        neg_candidates = [pid for pid in available_pids if pid not in pos_pids]
        neg_pids = random.sample(neg_candidates, min(self.num_negatives, len(neg_candidates)))
        neg_passages = [self.passages[pid]['text'] for pid in neg_pids]

        return {"query": query, "positive": pos_passage, 'negatives': neg_passages}
    

dataset = MSMARCO(queries, corpus, qrels)


# Tokenization
def collate_fn(batch, tokenizer, max_length=128):
    queries = [x['query'] for x in batch]
    positives = [x['positive'] for x in batch]
    negatives = []

    for x in batch:
        negatives.extend(x['negatives'])

    q_tok = tokenizer(
        queries, 
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    p_tok = tokenizer(
        positives + negatives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {"query": q_tok, "passages":p_tok}


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=training_batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda x: collate_fn(x, tokenizer)
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class DualEncoder(nn.Module):
    def __init__(self, query_model_name, passage_model_name):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.passage_encoder = AutoModel.from_pretrained(passage_model_name)

    def encode_query(self, **inputs):
        out = self.query_encoder(**inputs)
        return out.last_hidden_state[:, 0]        # CLS
    
    def encode_passage(self, **inputs):
        out = self.passage_encoder(**inputs)
        return out.last_hidden_state[:, 0]        # CLS 
    

model = DualEncoder(
    query_model_name="bert-base-uncased",
    passage_model_name="bert-base-uncased"
).to(device)


def contrastive_loss(q_emb:Tensor, p_emb:Tensor, temperature:float=dual_encoder_temp) -> Tensor:

    M = q_emb.shape[0]
    N = p_emb.shape[0]

    q_emb = F.normalize(q_emb, dim=-1)
    p_emb = F.normalize(p_emb, dim=-1)
    
    scores = torch.matmul(q_emb, p_emb.T)/temperature
    labels = torch.arange(M) * int(N/M)
    labels = labels.to(device=scores.device)

    loss = F.cross_entropy(scores, labels)
    return loss

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
train_loss = []

print('Training Beginning')
for i in range(epochs):

    model.train()
    epoch_loss = 0

    for step, batch in enumerate(dataloader):

        q_inputs = {k: v.to(device) for k, v in batch["query"].items()}
        p_inputs = {k: v.to(device) for k, v in batch["passages"].items()}

        q_emb = model.encode_query(**q_inputs)
        p_emb = model.encode_passage(**p_inputs)

        loss = contrastive_loss(q_emb, p_emb)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del q_emb, p_emb, loss  # Explicitly free memory
        torch.cuda.empty_cache()
        break

    avg_loss = epoch_loss/len(dataloader)
    train_loss.append(avg_loss)
    print('Epoch Completed')
    break

# plotting Loss
plt.figure(figsize=(12, 12))
plt.suptitle("Bi-Encoder Training Loss")

plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss", linestyle="-", marker="o")

plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.style.use('bmh')
plt.savefig("/work/mbouthil/MMATH-CM-Research-Project/RAG/figures/loss_curve" + save_name + ".png", dpi=300)


# Saving Encoder Weights
save_dir = "/work/mbouthil/MMATH-CM-Research-Project/RAG/model_weights"
model.query_encoder.save_pretrained(f"{save_dir}/query_encoder" + save_name)
model.passage_encoder.save_pretrained(f"{save_dir}/passage_encoder" + save_name)

tokenizer.save_pretrained(save_dir)

print('Training Complete')