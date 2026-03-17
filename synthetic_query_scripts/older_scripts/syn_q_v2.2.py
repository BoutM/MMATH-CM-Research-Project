import pandas as pd
import numpy as np
import random
import sys
import os
import json
import shutil
import random
from tqdm import tqdm
from torch import Tensor
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM
logging.set_verbosity_error()
sys.path.append('/mnt/hpc/work/mbouthil/MMATH-CM-Research-Project')
from packages.marco_dataloader import MSMARCO

test=False
save_name='syn_q_v2.2'

### Loading data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, _, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")


random.seed(42)
J=100_000
qrels = list(qrels.items())[:J]


system_prompt = '''You are a subject matter expert in your field with substantial accumulated knowledge in a
specific subject or topic, validated by academic degrees, certifications, and/or years of
professional experience in that field.

Write a question that is answered by the provided passage(s). Ensure that your question is concise and answered by the passage(s). 
Provide only the question and format it as follows:**question**. 
''' # Prompting Method 2

# Loading LM
from packages.llama import Llama_LM
llama = Llama_LM()

def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]
batches = batch_splits(qrels, batch_size=64)


syn_dict = dict()

for batch in batches:

    q_ids, p_data = zip(*batch)
    passages = []

    # Formatting passages for LLM message
    for entry in p_data:
        p_ids = list(entry.keys())
        p_ids = [key for key, value in entry.items() if value > 0]

        if len(p_ids) > 1:
            p_texts = []
            for i, p_id in enumerate(p_ids):
                p_texts.append(f"passage {i+1}: {corpus[p_id]['text']}")
            passages.append("\n".join(p_texts))

        else:
            passages.append(corpus[p_ids[0]]['text'])


    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": passage}
        ]
        for passage in passages
    ]

    syn_queries = llama.prompt(messages)

    for i, q_id in enumerate(q_ids):
        syn_dict[q_id] = syn_queries[i]

    if test == True:
        break

qrels = dict(qrels)


### Saving new dataset ###
# Creating directories
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = original_dir + "_" + save_name
os.makedirs(modified_dir, exist_ok=True)
os.makedirs(os.path.join(modified_dir, "qrels"), exist_ok=True)


# Copying old Corpus 
shutil.copy(
    f"{original_dir}/corpus.jsonl", 
    f"{modified_dir}/corpus.jsonl"
)


# Saving synthetic queries
queries_path = os.path.join(modified_dir, "queries.jsonl")
with open(queries_path, 'w') as f:
    for query_id, query_text in syn_dict.items():
        entry = {"_id": str(query_id), "text": query_text}
        f.write(json.dumps(entry) + '\n')


# Saving qrels
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")