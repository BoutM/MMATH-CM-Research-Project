# Creating additional Passages
import os
import json
import torch
import shutil
from dotenv import load_dotenv
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, logging, AutoModelForCausalLM
logging.set_verbosity_error()

### Pre ambles ###
save_name='_synp_1'
llm_temp=0.1
max_token=256
test=False

### Loading Data ###
data_dir = "/work/mbouthil/datasets/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")

# Gathering singular query passage mappings
pass_info = [(key, list(qrels[key].keys())[0], list(qrels[key].values())[0]) 
             for key in qrels.keys() if len(qrels[key]) < 2]

passages = [corpus[str(id)]['text'] for _, id, _ in pass_info]


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


### System instruction prompt ###
system_prompt='''
You are a helful AI assistant. 

You are to write a query for the provided passage. Provide only the new query.
'''


### LLM function ###
def llm_pass(
        messages:list[list[dict]],
        padding:bool=True,
        truncation:bool=True,
        max_tokens:int=max_token, 
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


def batch_splits(item:list, batch_size:int=64):

    for i in range(0, len(item), batch_size):
        yield item[i:i + batch_size]

batches = batch_splits(pass_info)



for batch in batches:

    max_id = max([int(key) for key in corpus.keys()])
    passages = [corpus[str(id)]['text'] for _, id, _ in batch]
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": passage}
        ]
        for passage in passages
    ]
    new_passages = llm_pass(messages)
    new_passages = [passage[11:] for passage in new_passages]

    for i, tuple in enumerate(batch):
        q_id, p_id, score = tuple
        qrels[q_id] = {p_id: score, str(max_id+i): score}
        corpus[max_id+i] = {'text': new_passages[i], 'title': ''}

    if test == True:
        break


### Writing new data ###
# Copying queries
original_dir = "/work/mbouthil/datasets/msmarco"
modified_dir = original_dir + save_name
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
        entry = {str(passage_id): data}
        f.write(json.dumps(entry) + '\n')


# Saving qrels + new qrels 
qrels_path = os.path.join(modified_dir, "qrels", "train.tsv")
with open(qrels_path, 'w') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            f.write(f"{str(query_id)}\t{str(doc_id)}\t{score}\n")