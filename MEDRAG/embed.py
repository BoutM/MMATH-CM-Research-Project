# Creation of the Passage Embedding Dataset

import pandas as pd
import torch
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import login
from transformers import AutoTokenizer, logging, AutoModel
logging.set_verbosity_error()
import numpy as np
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
import os
from torch import Tensor
import faiss
import json

# Loading Data
df = pd.read_csv("/work/mbouthil/projects/research_project/MEDRAG/synthetic_data/add_synq.csv")
passages = ['Subject ID: ' + str(df['SUBJECT_ID'].iloc[i]) + '\n'  + str(df['PASSAGE'].iloc[i]) for i in range(len(df))]

# Selecting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Passage Encoder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
passage_encoder = AutoModel.from_pretrained("bert-base-uncased")

passage_encoder = AutoModel.from_pretrained(
    "/work/mbouthil/projects/research_project/MEDRAG/model_weights/passage_encoder_3"
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
np.save('/work/mbouthil/projects/research_project/MEDRAG/retrieval_data/passage_embeddings_3.npy', embeddings_np)

# Creating FAISS Index
N, d = embeddings_np.shape
index = faiss.IndexFlatIP(d)
index.add(embeddings)
faiss.write_index(index, "/work/mbouthil/projects/research_project/MEDRAG/retrieval_data/passage_3.index")

# Creating FAISS metadata.json
with open("/work/mbouthil/projects/research_project/MEDRAG/retrieval_data/passage_metadata_3.jsonl", "w") as f:
    for idx, passage in enumerate(passages):
        record = {
            "id": idx,
            "text": passage
        }
        f.write(json.dumps(record) + "\n")

