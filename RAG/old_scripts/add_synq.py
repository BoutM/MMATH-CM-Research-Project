# Importing packages
import pandas as pd
import torch
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()
import random
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
import os


# Importing Data
df = pd.read_csv('/work/mbouthil/projects/research_project/MEDRAG/synthetic_data/synq.csv', nrows=10000)
queries = df['QUERY'].tolist()

# Loading LLM
load_dotenv('token.env')
token = os.getenv('HUGGINGFACE_TOKEN')

# Loading model - pass token directly
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    token=token  # Pass token here
)


system_prompt = '''
You are a helpful AI Assistant. You will be provided a question written by a doctor.

Your task is to rewrite this question 7 different ways. However, it is crucial that these alternative questions ask the same
underlying question as the provided question.
Feel free to use more or less medical terminology and medical acronyms where you see fit. 
Moreover, keep the questions relatively simple and straight forward. 

Format your output as follows:

**Query_1**

**Query_2**

...

**Query_7** 
'''

def llm_batch(notes: list[str], system_prompt:str) -> list[str]:
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

# Creating Batches
def chunked(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# Running Batches
add_queries = []

for batch in chunked(queries, 8):
    add_queries.extend(llm_batch(batch, system_prompt))

# Splitting Query outputs
def queries_split(queries:list[str]) -> list[list[str]]:

    queries = [re.split(r"\*\*Query_\d+\*\*\s*", query)[1:] for query in queries]
    queries = [[query.replace('/n', '').strip() for query in query_list if query.replace('/n', '').strip()] for query_list in queries]
    return queries


split_queries = queries_split(add_queries)

def df_expansion(df:pd.DataFrame, add_queries:list[list[str]]) -> pd.DataFrame:

    new_df = pd.DataFrame(columns=df.columns)

    for i in range(len(df)):
        queries = add_queries[i]
        row = df.iloc[[i]]

        if not queries:
            new_df = pd.concat([new_df, row], ignore_index=True)
            continue
        duplicates = pd.concat([row] * len(queries), ignore_index=True)
        duplicates.loc[:, "QUERY"] = queries

        new_rows = pd.concat([row, duplicates], ignore_index=True)
        new_df = pd.concat([new_df, new_rows], ignore_index=True)

    return new_df

new_df = df_expansion(df, split_queries)
new_df.to_csv("/work/mbouthil/projects/research_project/MEDRAG/synthetic_data/add_synq.csv", index=False)

print('Completed')