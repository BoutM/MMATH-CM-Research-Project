# Loading Packages
import pandas as pd
import numpy as np
import random
import os
import torch
import faiss
import json
import re
import shutil
import random
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM
logging.set_verbosity_error()
from beir.datasets.data_loader import GenericDataLoader
from dotenv import load_dotenv
from huggingface_hub import login

# Additional Synthetic Queries to be created
N=100_000


# Loading Data
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split='train')


# Creating Query Subset
q_subset = [(keys, values) for keys, values in queries.items()][:N]


# Loading LLM
load_dotenv('/work/mbouthil/MMATH-CM-Research-Project/token.env')
token = os.getenv('HUGGINGFACE_TOKEN')
login(token=token)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer =  AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    token=token
)


cot_prompt='''
You are a helpful AI assistant. You are to follow the following instructions:

You will be given a query. Your task is to think about how you would produce a new query asking the same underlying question.

Proivide your thoughts:
'''

creation_prompt=f'''
You are a helpful AI assistant. Your task is to create a new question from the provided query, asking the same underlying infromation. 

Provide only your new question.  

Moreover, consider the following:
'''

judge_prompt='''
You are a helpful AI assistant. 

Your task is to judge whether both provided queries pose the same underlying question. 

Ensure that your answer contains TRUE or FALSE. 
'''

### LLM function ###
def llm_pass(
        messages:list[list[dict]],
        padding:bool=True,
        truncation:bool=True,
        max_tokens:int=256, 
        temp:float=0.1,
        top_p:float=0.9,
) -> list[str]:

    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=padding,
        truncation=truncation
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            do_sample=True
        )

    responses = []
    for i in range(len(messages)):
        gen = outputs[i][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(gen, skip_special_tokens=True))

    return responses


### Creating Batches ###
def batch_splits(queries:list, batch_size:int=64) -> list[tuple[str, str]]:

    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]

batches = batch_splits(q_subset)

max_id = int(max([int(key) for key in queries.keys()]))
global_counter = max_id + 1 


for batch in batches:

    ids, old_queries = zip(*batch)
    N = len(old_queries)

    # Step 1
    cot_messages = [
        [
            {"role": "system", "content": cot_prompt},
            {"role": "user", "content": query}
        ]
        for query in old_queries
    ]
    cot_responses = llm_pass(cot_messages)
    cot_responses = [response[11:] for response in cot_responses]

    # Step 2
    creation_messages = [
        [
            {"role": "system", "content": creation_prompt + cot_responses[i]},
            {"role": "user", "content": query}
        ]
        for i, query in enumerate(old_queries)
    ]
    syn_queries = llm_pass(creation_messages)
    syn_queries = [query[11:] for query in syn_queries]

    # Step 3
    judge_messages = [
        [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": 
             str('"' + old_queries[i] + '"\n\nand \n' + syn_queries[i])
             }
        ]
        for i in range(N)
    ]
    verdicts = llm_pass(judge_messages)
    verdicts = [1 if 'TRUE' in answer else 0 for answer in verdicts]

    filtered = [(id, query) for id, query, verdict in zip(ids, syn_queries, verdicts) if verdict == 1]
    ids, syn_queries = [list(item) for item in zip(*filtered)]

    for i, id in enumerate(ids):
        new_id = global_counter
        qrels[new_id] = qrels[id]
        queries[new_id] = syn_queries[i]
        global_counter += 1



### Saving new dataset ###
# paths
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = original_dir + "_synq_2"


# Create directories
os.makedirs(modified_dir, exist_ok=True)
os.makedirs(os.path.join(modified_dir, "qrels"), exist_ok=True)


# Copy old Corpus 
shutil.copy(
    f"{original_dir}/corpus.jsonl", 
    f"{modified_dir}/corpus.jsonl"
)


# Saving queries with additional ones
queries_path = os.path.join(modified_dir, "queries.jsonl")
with open(queries_path, 'w') as f:
    for query_id, query_text in queries.items():
        entry = {"_id": str(query_id), "text": query_text}  # Force string
        f.write(json.dumps(entry) + '\n')


# Save qrels with additional ones
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")  # Force string