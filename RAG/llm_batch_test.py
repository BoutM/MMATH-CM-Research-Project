# Creating additional Passages
import os
import re
import gc
import sys
import json
import torch
import shutil
import time
import numpy as np
import tqdm as tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModelForCausalLM
logging.set_verbosity_error()


### Loading Data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")

query_data = [(key, value) for key, value in queries.items()]


### Loading LM Model ###
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


### Sysyem Prompt ###
system_prompt = '''
You are a helpful AI Assistant. You are to follow the following instructions:

You will be given a query. Your task is to create a new query that asks the same questions as
the provided query, however, posed differently. Provide only the new query and format it as follows:

**new query** 
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


def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]

batches = batch_splits(query_data)


batch_sizes =[32, 64, 128, 256, 512]
times = []

for size in batch_sizes:

    batches = batch_splits(query_data, batch_size=size)
    for batch in batches:

        max_id = max([int(key) for key in queries.keys()]) + 1
        q_ids, query_texts = zip(*batch)

        start_time = time.time()
        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            for query in query_texts
        ]
        new_queries = llm_pass(messages)
        new_queries = [
            re.findall(r'\*\*([^*]+)\*\*', query)[0] 
            if len(re.findall(r'\*\*([^*]+)\*\*', query)) != 0 
            else query
            for query in new_queries
            ]

        for i, q_id in enumerate(q_ids):
            new_id = max_id + i
            qrels[new_id] = qrels[q_id]
            queries[new_id] = new_queries[i]
        end_time = time.time()
        times.append((end_time - start_time)/size)
        print(f"Batch size {size} test completed in {end_time - start_time} seconds")
        break


# plotting Loss
plt.figure(figsize=(12, 12))
plt.suptitle("LM Generation Time vs Batch Size", fontsize=16)
plt.plot(batch_sizes, times, label="LM Gen Time per Prompt", linestyle="-", marker="o")
plt.ylabel("Seconds/Prompt")
plt.xlabel("Batch Size")
plt.xticks(batch_sizes)
plt.legend()
plt.style.use('bmh')
plt.savefig("/work/mbouthil/MMATH-CM-Research-Project/RAG/figures/LM_batch_size.png", dpi=300)