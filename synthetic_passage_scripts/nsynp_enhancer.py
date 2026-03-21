import os
import re
import gc
import sys
import json
import torch
import shutil
import random
import numpy as np
import tqdm as tqdm
from dotenv import load_dotenv
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModelForCausalLM
logging.set_verbosity_error()
sys.path.append('/mnt/hpc/work/mbouthil/MMATH-CM-Research-Project')
from packages.llama import Llama_LM


# This script adds negative passages to the desired dataset

K=200_000
old_dataset_name = 'syn_q_fs_agent_200k'
new_dataset_name = old_dataset_name+'-'

# Loading Data
data_dir = f"/work/mbouthil/datasets/msmarco_{old_dataset_name}"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
queries = dict(list(queries.items())[:K])


# Loading LM
llama = Llama_LM()


### System instruction prompt ###
system_prompt='''
You are a helful AI assistant. You are to follow the following instructions:

You will be given a question. Your task is to write a passage that does not answer the question. 

Provide only the new passage and format it as follows:

**new passage**
'''


def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]

batches = batch_splits(list(queries.items()))

# Creating Hard Negatives
max_id = max([int(key) for key in corpus.keys()]) +1 
for batch in batches:

    q_ids, q_text = zip(*batch)
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"}
        ]
        for query in q_text
    ]
    new_passages = llama.prompt(messages)

    for i, q_id in enumerate(q_ids):
        new_pid = max_id + i
        qrels[q_id] = {**qrels[q_id], 
                       **{str(new_pid): -1}}
        corpus[str(new_pid)] = {'text': new_passages[i], 'title': 'Negative synthetic passage'}

    max_id += len(batch)



### Saving new dataset ###
# Creating directories
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = f"{original_dir}_{new_dataset_name}"
os.makedirs(modified_dir, exist_ok=True)
os.makedirs(os.path.join(modified_dir, "qrels"), exist_ok=True)

# Copying queries
shutil.copy(
    f"{original_dir}/queries.jsonl", 
    f"{modified_dir}/queries.jsonl"
)


# Creating corpus with new passages
corpus_path = os.path.join(modified_dir, "corpus.jsonl")
with open(corpus_path, 'w') as f:
    for passage_id, data in corpus.items():
        entry = {"_id": str(passage_id), **data}
        f.write(json.dumps(entry) + '\n')


# Creating updated qrels
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")


print(f"{new_dataset_name} data creation complete")