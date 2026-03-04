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
import time
from tqdm import tqdm
from torch import Tensor
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModel, AutoModelForCausalLM


### Pre ambles ###
save_name = 'syn_p_agent_full'
llm_temp=0.1
test=False
batch_size=128


### Loading Data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split='train')

# Gathering singular query passage mappings
pass_info = [(key, list(qrels[key].keys())[0], list(qrels[key].values())[0]) 
             for key in qrels.keys() if len(qrels[key]) < 2]


### Loading LLM ###
# Authenticating Token
load_dotenv('/work/mbouthil/MMATH-CM-Research-Project/token.env')
token = os.getenv('HUGGINGFACE_TOKEN')

# Loading model and Tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer =  AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    token=token
)

### Sysyem Prompts ###
cot_prompt='''
You are a helpful AI assistant. 

You will be given a passage. Your task is to provide your thoughts regarding how you would rewrite this passage.

Proivide your thoughts:
'''

creation_prompt=f'''
You are a helpful AI assistant. 

Your task is to create a new passage from the provided one. Your new rewritten passage must contain the same infromation as the original. 

Provide only the new passage, formatted as: **new passage**

The following contains thoughts regarding how to rewrite the passage. Use these thoughts to guide your creation of a new passage.
\n
'''

judge_prompt='''
You are a helpful AI assistant. 

Your task is to judge whether both provided passages contain the same information.

If both passages contain the same information, ensure your response contains "TRUE", else ensure your response contains "FALSE".
'''


### LLM function ###
def llm_pass(
        messages:list[list[dict]],
        padding:bool=True,
        truncation:bool=True,
        max_tokens:int=500, 
        temp:float=llm_temp,
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
def batch_splits(queries:list, batch_size:int=64) -> list:
    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]

batches = batch_splits(pass_info, batch_size=batch_size)


### Creating new passsages ###
N = len(pass_info)
total_iters = np.ceil(N/batch_size)
iter_counter = 0

print(f"Creating {len(pass_info)} agnetic passages...")
start = time.time()
for batch in batches:

    max_id = int(max([int(key) for key in corpus.keys()]))
    q_ids, p_ids = zip(*[(q_id, p_id) for q_id, p_id, _ in batch])
    passages = [corpus[str(p_id)]['text'] for p_id in p_ids]
    N = len(passages)

    # Step 1
    cot_messages = [
        [
            {"role": "system", "content": cot_prompt},
            {"role": "user", "content": passage}
        ]
        for passage in passages
    ]
    cot_responses = llm_pass(cot_messages, max_tokens=100)
    cot_responses = [response[11:] for response in cot_responses]
    del cot_messages

    # Step 2
    creation_messages = [
        [
            {"role": "system", "content": creation_prompt + cot_responses[i]},
            {"role": "user", "content": passage}
        ]
        for i, passage in enumerate(passages)
    ]
    syn_passages = llm_pass(creation_messages, max_tokens=500)
    del cot_responses, creation_messages

    syn_passages = [
        re.findall(r'\*\*([^*]+)\*\*', passage)[0] 
        if len(re.findall(r'\*\*([^*]+)\*\*', passage)) != 0 
        else passage
        for passage in syn_passages
        ]

    # Step 3
    judge_messages = [
        [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": 
             str('"' + passages[i] + '"\n\nand \n' + syn_passage)
             }
        ]
        for i, syn_passage in enumerate(syn_passages)
    ]
    verdicts = llm_pass(judge_messages, max_tokens=25)
    verdicts = [1 if 'TRUE' in answer else 0 for answer in verdicts]

    filtered = [(q_id, p_id, passage) 
                for q_id, p_id, passage, verdict in zip(q_ids, p_ids, syn_passages, verdicts) if verdict == 1]

    for i, tuple in enumerate(filtered):
        q_id, p_id, passage = tuple
        qrels[q_id] = {p_id: 1, str(max_id+i): 1}
        corpus[max_id+i] = {'text': passage, 'title': 'Synthetic agentic passage'}

    if test == True:
        break

    iter_counter += 1
    del judge_messages, verdicts, syn_passages
    torch.cuda.empty_cache()
    print(f"Iteration {iter_counter} of {int(total_iters)} complete")
end = time.time()

print("\n")
print("Synthetic agentic passasges created")
print("\n")
print(f"Execution time: {(end-start)/60} minutes")
print("\n")

### Writing new data ###
# Copying queries
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = f"{original_dir}_{save_name}"
os.makedirs(modified_dir, exist_ok=True)
os.makedirs(os.path.join(modified_dir, "qrels"), exist_ok=True)

shutil.copy(
    f"{original_dir}/queries.jsonl", 
    f"{modified_dir}/queries.jsonl"
)


# Saving passages + new passages
corpus_path = os.path.join(modified_dir, "corpus.jsonl")
with open(corpus_path, 'w') as f:
    for passage_id, data in corpus.items():
        entry = {"_id": str(passage_id), **data}
        f.write(json.dumps(entry) + '\n')


# Saving qrels + new qrels 
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")


print(f"{save_name} data creation complete")