# Creation of Additional Synthetic Queries

import torch
from transformers import AutoTokenizer, logging, AutoModelForCausalLM
logging.set_verbosity_error()
import json
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm
from dotenv import load_dotenv
import re
import json
import os
import shutil

# Additional Synthetic Queries to be created:
N=100_000


### Loading data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
t_corpus = corpus.copy()
t_queries = queries.copy()
t_qrels = qrels.copy()

max_id = int(max([int(key) for key in t_queries.keys()]))
q_subset = [(keys, values) for keys, values in queries.items()][:N]


### Loading LLM ###

# Authenticating Token
load_dotenv('token.env')
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


### Sysyem Prompt ###
system_prompt = '''
You are a helpful AI Assistant. You are to follow the following instructions:

You will be given a query. Your task is to create a new query that asks the same questions as
the provided query, however, posed differently. Provide only the new query and format is as follows:

**new query** 
'''


### LLM function ###
def llm_call(notes: list[str], system_prompt:str) -> list[str]:
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": note}
        ]
        for note in notes
    ]

    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True
        )

    responses = []
    for i in range(len(notes)):
        gen = outputs[i][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(gen, skip_special_tokens=True))

    return responses


### Creating Batches ###
def batch_splits(queries:list, batch_size:int=64):
    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]

batches = batch_splits(q_subset)


### Creating new queries ###
new_queries = []
old_ids = []

for batch in batches:

    ids = [id[0] for id in batch]
    queries = [id[1] for id in batch]

    queries = llm_call(queries, system_prompt)
    queries = [
        re.findall(r'\*\*([^*]+)\*\*', query)[0] 
        if len(re.findall(r'\*\*([^*]+)\*\*', query)) != 0 
        else query 
        for query in queries
        ]
    new_queries.extend(queries)
    old_ids.extend(ids)



### Expanding Dataset ###
new_q_id = max_id + 1 

for i, id in enumerate(old_ids):

    t_queries[new_q_id] = new_queries[i]            # Creating new query
    t_qrels[new_q_id] = t_qrels[id]                 # Mapping new query to positive passages
    new_q_id += 1


### Saving new dataset ###
# paths
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = original_dir + "_synq_1"

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
    for query_id, query_text in t_queries.items():
        entry = {"_id": str(query_id), "text": query_text}  # Force string
        f.write(json.dumps(entry) + '\n')


# Save qrels with additional ones
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in t_qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")  # Force string