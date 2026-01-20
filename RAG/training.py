import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, AutoModel
logging.set_verbosity_error()
import random
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Loading data
df = pd.read_csv("/work/mbouthil/projects/research_project/MEDRAG/synthetic_data/add_synq.csv")

queries = ['Subject ID: ' + str(df['SUBJECT_ID'].iloc[i]) + '\n'  + str(df['QUERY'].iloc[i]) for i in range(len(df))]
passages = ['Subject ID: ' + str(df['SUBJECT_ID'].iloc[i]) + '\n'  + str(df['PASSAGE'].iloc[i]) for i in range(len(df))]


class QPDataset(Dataset):
    def __init__(self, queries, passages):
        self.queries = queries
        self.passages = passages

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "passage": self.passages[idx]
        }
    

dataset = QPDataset(queries, passages)
dataloader = DataLoader(
    dataset,
    batch_size=32,   # BATCH SIZE IMPORTANT
    shuffle=True
)


class BiEncoder(nn.Module):
    def __init__(self, query_model_name, passage_model_name):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.passage_encoder = AutoModel.from_pretrained(passage_model_name)

    def encode_query(self, **inputs):
        out = self.query_encoder(**inputs)
        return out.last_hidden_state[:, 0]  # CLS

    def encode_passage(self, **inputs):
        out = self.passage_encoder(**inputs)
        return out.last_hidden_state[:, 0]  # CLS
    

def contrastive_loss(q_emb, p_emb, temperature=1.0):

    """
    q_emb: (B, D)
    p_emb: (B, D)
    """
    
    # Similarity matrix: (B, B)
    scores = torch.matmul(q_emb, p_emb.T) / temperature
    labels = torch.arange(scores.size(0)).to(scores.device)

    loss = F.cross_entropy(scores, labels)
    return loss


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(texts, max_length=256):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiEncoder(
    query_model_name="bert-base-uncased",
    passage_model_name="bert-base-uncased"
).to(device)

# Selecting Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

num_epochs = 50
train_loss = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in dataloader:  # batch["query"], batch["passage"]
        queries = batch["query"]
        passages = batch["passage"]

        q_inputs = tokenize_batch(queries)
        p_inputs = tokenize_batch(passages)

        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        p_inputs = {k: v.to(device) for k, v in p_inputs.items()}

        q_emb = model.encode_query(**q_inputs)
        p_emb = model.encode_passage(**p_inputs)

        q_emb = F.normalize(q_emb, dim=1)
        p_emb = F.normalize(p_emb, dim=1)

        loss = contrastive_loss(q_emb, p_emb)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(dataloader)
    train_loss.append(avg_loss)


plt.figure(figsize=(12, 12))
plt.suptitle("Bi-Encoder Training Loss")

plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss", linestyle="-", marker="o")

plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.style.use('bmh')
plt.savefig("/work/mbouthil/projects/research_project/MEDRAG/figures/loss_curve_3.png", dpi=300)

# Saving Encoder Weights
save_dir = "/work/mbouthil/projects/research_project/MEDRAG/model_weights"
model.query_encoder.save_pretrained(f"{save_dir}/query_encoder_3")
model.passage_encoder.save_pretrained(f"{save_dir}/passage_encoder_3")

tokenizer.save_pretrained(save_dir)