# Training of the Dual Encoders using contrastive loss
import json
import random
import torch
import os
import gc
import sys
import shutil
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from torch.cuda.amp import autocast, GradScaler
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, AutoModel
sys.path.append('/mnt/hpc/work/mbouthil/MMATH-CM-Research-Project')
from packages.marco_dataloader import MSMARCO

save_name='syn_q_sagent'
test=False
batch_size=64


### Loading data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, _, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")


# Gathering 500_000 random qrels
random.seed(42)
J=500_000
qrels = random.sample(list(qrels.items()), J)


# Loading LM
from packages.llama import Llama_LM
llama = Llama_LM()


### Agent Prompts ###
creation_prompt = '''You are a subject matter expert in your field with substantial accumulated knowledge in a
specific subject or topic, validated by academic degrees, certifications, and/or years of
professional experience in that field.

Write a single question that elaborates on the provided passage(s). This question must be answered by the passage(s). 

Provide only the question and format it as follows:**question**.
'''

judge_prompt='''You are a subject matter expert in your field with substantial accumulated knowledge in a
specific subject or topic, validated by academic degrees, certifications, and/or years of
professional experience in that field.

Your task is to judge whether the provided question is answered by the provided passage(s). If there are multiple passages,
the question have an answer contained within each passage. 

If the question satisfies the condition, ensure your response contains "TRUE". Otherwise, ensure your response contains "FALSE".
'''


# Creating Batches
def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]
batches = batch_splits(qrels, batch_size=batch_size)


syn_dict=dict()
new_qrels=dict()
qrels=dict(qrels)
counter=0

for batch in batches:
    q_ids, p_data = zip(*batch)
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

    creation_messages = [
        [
            {"role": "system", "content": creation_prompt},
            {"role": "user", "content": f"{passage}"}
        ]
        for passage in passages
    ]
    syn_queries = llama.prompt(creation_messages)
    del creation_messages

    judge_messages = [
        [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": f"Question: {syn_query}\nPassage(s):\n{passages[i]}"}
        ]
        for i, syn_query in enumerate(syn_queries)
    ]

    verdicts = llama.prompt(judge_messages, da_wrap=False)
    verdicts = [1 if 'TRUE' in answer else 0 for answer in verdicts]

    filtered = [(q_id, syn_query) for q_id, syn_query, verdict in zip(q_ids, syn_queries, verdicts) if verdict == 1]

    for q_id, syn_query  in filtered:
        syn_dict[q_id] = syn_query
        new_qrels[q_id] = qrels[q_id]


    counter += len(filtered)
    if counter >= 100_000:
        break
    if test==True:
        break



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
    for query_id, doc_scores in new_qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")
