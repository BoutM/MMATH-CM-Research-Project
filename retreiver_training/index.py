import os
import gc
import json
import sys
import torch
import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
import torch.nn.functional as F
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from transformers import AutoTokenizer, AutoModel


'''
This script is to be used in the event that OOM/memory issues occurs after training within the training.py script.
'''

write_embeddings=True
model_name = "r400s_sq-v2"

corpus_path = "/work/mbouthil/datasets/msmarco/corpus.jsonl"
output_dir = "/work/mbouthil/MMATH-CM-Research-Project/retreiver_training/retrieval_data"
embeddings_path = f"{output_dir}/embeddings_{model_name}.npy"
device = "cuda" if torch.cuda.is_available() else "cpu"

N = 8_841_823       # Corpus entries
d = 768             # BERT embedding dimension


if write_embeddings==True:
    def stream_msmarco_chunks(path, chunk_size=5000):
        buffer = []
        with open(path, "r") as f:
            for line in f:
                doc = json.loads(line)
                buffer.append((doc["_id"], doc["text"]))
                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []
        if buffer:
            yield buffer


    ### Loading Passage Encoder ###
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    passage_encoder = AutoModel.from_pretrained(
        f"/work/mbouthil/MMATH-CM-Research-Project/retreiver_training/model_weights/passage_encoder_{model_name}"
        ).to(device)
    passage_encoder.eval()


    ### Creating memmap ###
    embedding_memmap = np.memmap(
        embeddings_path,
        dtype='float32',
        mode='w+',
        shape=(N, d)
    )
    batch_size = 64
    chunksize=5000
    offset = 0


    ### Writting Embeddings ###
    print('Writing Embeddings')
    pbar=tqdm(total=np.ceil(N/chunksize), file=sys.stdout)
    for chunk in stream_msmarco_chunks(corpus_path, chunksize):

        for i in range(0, len(chunk), batch_size):
            batch = chunk[i : i + batch_size]
            passages = [x[1] for x in batch]
            
            with torch.no_grad():
                inputs = tokenizer(
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)

                out = passage_encoder(**inputs)
                last_hidden_state = out.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                emb = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                emb = F.normalize(emb, p=2, dim=-1)
                emb = emb.float().cpu().numpy()
                emb = np.ascontiguousarray(emb, dtype=np.float32)
                # faiss.normalize_L2(emb)  # Keep this one
                # index.add(emb)           # Add immediately after normalizing

                embedding_memmap[offset:offset + emb.shape[0]] = emb
                offset += emb.shape[0]

                del inputs, emb

            if i % (batch_size*10) == 0:
                torch.cuda.empty_cache()

        del chunk
        pbar.update(1)
        gc.collect()
    pbar.close()
    print('Embeddings written')

    embedding_memmap.flush()
    del passage_encoder, tokenizer, embedding_memmap
    gc.collect()
    torch.cuda.empty_cache()
    print('Embeddings written\n')


### Reading memmap ###
embeddings = np.memmap(
    embeddings_path,
    dtype='float32',
    mode='r',
    shape=(N, d)
)


### Writting index ###
print("Writting index")
index = faiss.IndexFlatIP(d)
chunk_size = 100_000

for start_idx in range(0, N, chunk_size):
    end_idx = min(start_idx + chunk_size, N)
    chunk_emb = embeddings[start_idx:end_idx]
    index.add(chunk_emb)
    del chunk_emb
    gc.collect()

faiss.write_index(index, f"{output_dir}/{model_name}.index")

print("Index written")