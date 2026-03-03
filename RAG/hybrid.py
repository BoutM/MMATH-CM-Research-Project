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

test=False
save_name='synq_v2_2'

### Loading data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, _, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")


random.seed(42)
J=100_000
qrels = random.sample(list(qrels.items()), J)


system_prompt = '''You are a subject matter expert in your field with substantial accumulated knowledge in a
specific subject or topic, validated by academic degrees, certifications, and/or years of
professional experience in that field.s

Write a question that elaborates on the provided passage(s). Ensure that your question is answered by the passage(s). 
Provide only the question and format it as follows:**question**. 
'''

# system_prompt = '''You are a subject matter expert in your field with substantial accumulated knowledge in a
# specific subject or topic, validated by academic degrees, certifications, and/or years of
# professional experience in that field.

# Write a question that is answered by the provided passage(s). Ensure that your question is concise and answered by the passage(s). 
# Provide only the question and format it as follows:**question**. 
# '''

# Loading LM
from packages.llama import Llama_LM
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

    if test == True:
        break

qrels = dict(qrels)
