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
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM
logging.set_verbosity_error()


# New dataset name
dataset_name = 'syn_q_agent_full'
batch_size=64
test=False

# Loading Data
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split='train')


### Loading LLM ###
# Authenticating Token
load_dotenv('/work/mbouthil/MMATH-CM-Research-Project/token.env')
token = os.getenv('HUGGINGFACE_TOKEN')

# Loading model and model tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer =  AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    token=token
)


### Agent Prompts ###
cot_prompt='''
You are a helpful AI assistant. 

You will be given a query. Your task is to provide your thoughts regarding how you would rewrite this query, in the same style, and asking the same underlying question.

Proivide your thoughts:
'''

creation_prompt=f'''
You are a helpful AI assistant. 

Your task is to create a new query from the provided one. Your new rewritten query must ask the same underlying question, and attempt to match the way the question (query) is posed. 

Provude only the new query and format it as follows: **new_query**

The following contains thoughts regarding how to rewrite the query. Use these thoughts to guide your creation of a new query.
\n
'''

judge_prompt='''
You are a helpful AI assistant. 

Your task is to judge whether both provided queries pose the same underlying question, and are posed similarly. 

If both queries satisfies the above, ensure your response contains "TRUE", else ensure your response contains "FALSE".
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
def batch_splits(queries:list[tuple], batch_size:int=64) -> list[tuple]:
    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]


### Creating new queries ###
query_list = [(keys, values) for keys, values in queries.items()]
batches = batch_splits(query_list, batch_size=batch_size)
N = len(query_list)
total_batches = np.ceil(N/batch_size)
batch_counter = 0

print("Creating synthetic queries...")
for batch in batches:

    ids, old_queries = zip(*batch)
    N = len(old_queries)

    # LLM Agent #1 - CoT
    cot_messages = [
        [
            {"role": "system", "content": cot_prompt},
            {"role": "user", "content": query}
        ]
        for query in old_queries
    ]
    cot_responses = llm_pass(cot_messages)
    cot_responses = [response[11:] for response in cot_responses]
    del cot_messages

    # LLM Agent #2 - New Query Creation
    creation_messages = [
        [
            {"role": "system", "content": creation_prompt + cot_responses[i]},
            {"role": "user", "content": query}
        ]
        for i, query in enumerate(old_queries)
    ]
    syn_queries = llm_pass(creation_messages)
    del cot_responses, creation_messages
    
    syn_queries = [
        re.findall(r'\*\*([^*]+)\*\*', syn_query)[0] 
        if len(re.findall(r'\*\*([^*]+)\*\*', syn_query)) != 0 
        else syn_query
        for syn_query in syn_queries
        ]

    # LLM Agent #3 - Judge
    judge_messages = [
        [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": 
             str('"' + old_queries[i] + '"\n\nand \n' + syn_query)
             }
        ]
        for i, syn_query in enumerate(syn_queries)
    ]
    verdicts = llm_pass(judge_messages)
    verdicts = [1 if 'TRUE' in answer else 0 for answer in verdicts]

    # Filtering by verdict
    filtered = [(id, syn_query) for id, syn_query, verdict in zip(ids, syn_queries, verdicts) if verdict == 1]
    max_id = max([int(key) for key in queries.keys()]) +1

    # Extending dataset
    for i, tuple in enumerate(filtered):
        id, syn_query = tuple
        new_id = max_id + i
        qrels[new_id] = qrels[id]
        queries[new_id] = syn_query

    batch_counter += 1
    del judge_messages, verdicts, syn_queries
    torch.cuda.empty_cache()
    print('f{batch_counter} of {total_batches} batches complete')
    if test == True:
        break



### Saving new dataset ###
# Creating directories
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = original_dir + "_" + dataset_name
os.makedirs(modified_dir, exist_ok=True)
os.makedirs(os.path.join(modified_dir, "qrels"), exist_ok=True)


# Copying old Corpus 
shutil.copy(
    f"{original_dir}/corpus.jsonl", 
    f"{modified_dir}/corpus.jsonl"
)


# Saving queries
queries_path = os.path.join(modified_dir, "queries.jsonl")
with open(queries_path, 'w') as f:
    for query_id, query_text in queries.items():
        entry = {"_id": str(query_id), "text": query_text}
        f.write(json.dumps(entry) + '\n')


# Saving qrels
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")

print('Data Saved')