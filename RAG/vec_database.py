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

# Loading Data
data_dir = "/work/mbouthil/projects/research_project/RAG/datasets/msmarco"
corpus, _, _ = GenericDataLoader(data_folder=data_dir).load(split="train")
passages = [corpus[doc_id]["text"] for doc_id in corpus]


# Selecting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading Passage Encoder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
passage_encoder = AutoModel.from_pretrained(
    "/work/mbouthil/projects/research_project/RAG/model_weights/passage_encoder"
).to(device)
passage_encoder.eval()


# Custom MSMARCO class
class MSMARCO:
    def __init__(self, passages):
    
        '''
        DataLoader for Vector Database
        '''

        self.passages = passages
    
    def __len__(self):
        return len(self.passages)
    
    def __getitem__(self, idx):
        passage = self.passages[idx]
        return {"passage": passage}


dataset = MSMARCO(passages)


def collate_fn(batch:int, tokenizer:object, max_length:int=128) -> dict:

    passages = [x['passage'] for x in batch]
    p_tok = tokenizer(passages, 
                      padding=True,
                      truncation=True, 
                      max_length=max_length,
                      return_tensors='pt'
                      )
    return {'passages': p_tok}


# Dataloader
dataloader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)
                        #collate_fn=lambda x:collate_fn(x, tokenizer))



dataloader = DataLoader(dataset, batch_size=64)


index = faiss.IndexFlatIP(768)
meta_file = open("/work/mbouthil/projects/research_project/RAG/retrieval_data/passage_metadata.jsonl", "w")
global_idx = 0

for batch in dataloader:
    with torch.no_grad():
        inputs = tokenizer(
            batch['passage'],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        emb = passage_encoder(**inputs).last_hidden_state[:, 0]

    emb = emb.float().cpu().numpy()
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    faiss.normalize_L2(emb)
    index.add(emb)

    for i, passage in enumerate(batch['passage']):
        meta = {
            'passage': passage,
            'idx': global_idx
        }
        meta_file.write(json.dumps(meta) + "\n")
        global_idx += 1

    del emb, inputs
    torch.cuda.empty_cache()

faiss.write_index(index, "/work/mbouthil/projects/research_project/RAG/retrieval_data/passage.index")



# # Creating embeddings
# embeddings = encode_passage(passages)

# # Moving to CPU
# embeddings_np = embeddings.cpu().numpy().astype("float32")
# np.save('/work/mbouthil/projects/research_project/RAG/retrieval_data/passage_embeddings.npy', embeddings_np)

# # Creating FAISS Index
# N, d = embeddings_np.shape
# index = faiss.IndexFlatIP(d)
# index.add(embeddings)
# faiss.write_index(index, "/work/mbouthil/projects/research_project/RAG/retrieval_data/passage.index")

# # Creating FAISS metadata.json
with open("/work/mbouthil/projects/research_project/RAG/retrieval_data/passage_metadata.jsonl", "w") as f:
    for idx, passage in enumerate(passages):
        record = {
            "id": idx,
            "text": passage
        }
        f.write(json.dumps(record) + "\n")

