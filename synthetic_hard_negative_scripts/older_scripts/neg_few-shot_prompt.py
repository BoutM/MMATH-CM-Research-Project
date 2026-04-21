import os
import sys
import json
import shutil
import random
import numpy as np
import tqdm as tqdm
from dotenv import load_dotenv
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
random.seed(42)


J=8
data_dir = f"/work/mbouthil/datasets/msmarco" 
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
rcorpus = dict(random.sample(list(corpus.items()), 2_000_000))

tokenized_corpus = [passage['text'].split() 
                    for passage 
                    in rcorpus.values()]
bm25 = BM25Okapi(tokenized_corpus)
del tokenized_corpus

# Begin by gathering examples:
N = len(qrels)
examples = []

for q_id, p_data in list(qrels.items())[N-J:]:

    query = queries[q_id]
    p_ids = list(p_data.keys())
    passage = corpus[random.choice(p_ids)]['text']
    examples.append((query, p_ids, passage))

del qrels, queries

system_prompt ='''
You are a helful AI assistant. You are to follow the following instructions:

You will be given a question and passage(s). Your task is to write a new passage that does not answer the question.
However, this new passage that you will create must be related to the themes of the question and passage(s). 

Provide only the new passage and format it as follows:

**new passage**

Consider the following examples:

{fs_examples}
'''

fs_examples =[]

for query, p_ids, passage in examples:

    query = query.split()
    n = len(p_ids) + 1
    p_ids_set = set(p_ids)

    bm25_scores = bm25.get_scores(query)
    top_indices = np.argsort(bm25_scores)[::-1][:n]
    
    for idx in top_indices:
        p_id, p_texts = list(rcorpus.items())[idx]
        if str(p_id) not in p_ids_set:
            neg_passage = p_texts['text']
            break

    fs_examples.append(f"Question: {" ".join(query)}\nPassage: {passage}\nNew passage: **{neg_passage}**")

del bm25, corpus

system_prompt=system_prompt.format(fs_examples="\n\n".join(fs_examples))

with open("/work/mbouthil/MMATH-CM-Research-Project/synthetic_passage_scripts/fs_sys_prompt.txt", "w") as f:
    f.write(system_prompt)