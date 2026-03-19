import pandas as pd
import numpy as np
import sys
import os
import json
import shutil
from dotenv import load_dotenv
from huggingface_hub import login
from beir.datasets.data_loader import GenericDataLoader
from transformers import logging
logging.set_verbosity_error()
sys.path.append('/mnt/hpc/work/mbouthil/MMATH-CM-Research-Project')
from packages.llama import Llama_LM


'''
This script creates K zero-shot synthetic queries based on the corresponding qrels passages. 
'''
K=200_000

### Loading data ###
save_name='syn_q_200k'
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, _, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
qrels = list(qrels.items())[:K]


system_prompt = '''You are a subject matter expert in your field with substantial accumulated knowledge in a
specific subject or topic, validated by academic degrees, certifications, and/or years of
professional experience in that field.

Write a question that elaborates on the provided passage(s). Ensure that your question is answered by the passage(s). 
Provide only the question and format it as follows:**question**. 
''' 


# Loading LM
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


### Saving new dataset ###
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = original_dir + "_" + save_name
os.makedirs(modified_dir, exist_ok=True)
os.makedirs(os.path.join(modified_dir, "qrels"), exist_ok=True)

shutil.copy(                                                        # Copying old Corpus 
    f"{original_dir}/corpus.jsonl", 
    f"{modified_dir}/corpus.jsonl"
)

queries_path = os.path.join(modified_dir, "queries.jsonl")          # Saving synthetic queries
with open(queries_path, 'w') as f:
    for query_id, query_text in syn_dict.items():
        entry = {"_id": str(query_id), "text": query_text}
        f.write(json.dumps(entry) + '\n')

qrels = dict(qrels)
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")       # Saving qrels
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")