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


'''This script changes the prompt engineering of the enhancer v2 script'''


K=200_000
test=False
old_dataset_name = 'syn_q_200k'
new_dataset_name = old_dataset_name+'-v3'

# Loading Data
data_dir = f"/work/mbouthil/datasets/msmarco_{old_dataset_name}"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
print(len(qrels))
qrels = list(qrels.items())[:K]


# Loading LM
llama = Llama_LM()


### System instruction prompt ###
system_prompt='''
You are a subject matter expert in your field with substantial accumulated knowledge in a
specific subject or topic, validated by academic degrees, certifications, and/or years of
professional experience in that field.

You will be given a question and passage(s). Your task is to write a new passage that does not answer the question.
However, this new passage that you will create is to use be related to the themes of the question and passage(s). 

Provide only the new passage and format it as follows:

**new passage**
'''


def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]

batches = batch_splits(qrels)
qrels=dict(qrels)

# Creating Hard Negatives
max_id = max([int(key) for key in corpus.keys()]) +1 
for batch in batches:

    q_ids, p_data = zip(*batch)
    questions = [queries[q_id] for q_id in q_ids]

    passages = []
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
            {"role": "user", "content": f"Question: {question}\nPassage(s):\n{passages[i]}"}
        ]
        for i, question in enumerate(questions)
    ]
    new_passages = llama.prompt(messages)

    for i, q_id in enumerate(q_ids):
        new_pid = max_id + i
        qrels[q_id] = {**qrels[q_id], 
                       **{str(new_pid): -1}}
        corpus[str(new_pid)] = {'text': new_passages[i], 'title': 'Negative synthetic passage'}

    max_id += len(batch)
    if test==True:
        break

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