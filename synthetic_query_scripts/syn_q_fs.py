# Loading Packages
import random
import os
import json
import shutil
import random
import sys
import gc
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from beir.datasets.data_loader import GenericDataLoader
sys.path.append('/mnt/hpc/work/mbouthil/MMATH-CM-Research-Project')
from packages.llama import Llama_LM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


'''
This script creates K synthetic queries using few-shot prompting
'''

test=False
save_name='syn_q_fs'
batch_size=64
K=100_000                               # Currently set to K=len(qrels)-8 for job 40046


### Loading data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")


### Gathering 8 exmaples ###
random.seed(39)
mulit_qrels = random.sample([(key, value) for key, value in qrels.items() if len(value) > 1],3)
single_qrels = random.sample([(key, value) for key, value in qrels.items() if len(value)==1], 5)
example_qrels=mulit_qrels+single_qrels

# Removing few shot example qrels from data
qrels = [(key, value) for key, value in qrels.items() if key not in dict(example_qrels).keys()]
qrels=qrels[:K]


system_prompt = '''You are a subject matter expert in your field with substantial accumulated knowledge in a
specific subject or topic, validated by academic degrees, certifications, and/or years of
professional experience in that field.

Write a question that is concise and directly answered by the provided passage(s).
Provide only the question and format it as follows: **question**

Here are some examples:

{examples}
'''

# Formating Examples:
examples=[]
for qrel in example_qrels:
    q_id, p_dict = qrel
    p_ids = [key for key, value in p_dict.items() if value > 0]

    query=queries[q_id]
    passages=[corpus[p_id]['text'] for p_id in p_ids]
    passages = "\n".join([f"Passage {i+1}: {passage}" for i, passage in enumerate(passages)])
    
    examples.append(f"{passages}\nQuestion: {query}")

system_prompt=system_prompt.format(examples="\n\n".join(examples))
del queries


def batch_splits(item:list, batch_size:int=batch_size):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]
batches = batch_splits(qrels, batch_size=64)


syn_dict=dict()
llama = Llama_LM()


for batch in batches:

    q_ids, p_data = zip(*batch)
    passages = []

    # Formatting passages for LLM message
    for entry in p_data:
        p_ids = list(entry.keys())
        p_ids = [key for key, value in entry.items() if value > 0]
        passages.append([f"passage {i+1}: {corpus[p_id]['text']}" 
                            for i, p_id in enumerate(p_ids)])

    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": passage}
        ]
        for passage in passages
    ]

    syn_queries = llama.prompt(messages)
    del passages, messages
    gc.collect()

    for i, q_id in enumerate(q_ids):
        syn_dict[q_id] = syn_queries[i]

    if test == True:
        break

qrels=dict(qrels)
qrels=dict([(key, value) for key, value in qrels.items() if key in syn_dict.keys()])


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
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")