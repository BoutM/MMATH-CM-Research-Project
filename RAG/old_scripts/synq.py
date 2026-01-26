# Synthetic Query Generation

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
from dotenv import load_dotenv
import os


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

# Desired rows to load in from NOTEEVENTS.csv dataset
nrow=1000

# loading only 100 rows of data
df = pd.read_csv("/work/mbouthil/projects/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz", nrows=nrow)
notes = df['TEXT'].tolist()


def chunk_by_sentence(text: str, max_chars: int=400) -> list[str]:
    
    """
    Split text into chunks of at most max_chars characters,
    preserving sentence boundaries.

    Parameters
    ----------
    text : str
        Input text to be chunked.
    max_chars : int
        Target maximum character length per chunk.

    Returns
    -------
    List[str]
        List of text chunks.
    """

    # 1. Split text into sentences (keeps punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Edge case: single sentence longer than max_chars
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence.strip())
            continue

        # Try to append sentence to current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


with ThreadPoolExecutor() as executor:
    print("Splitting Notes")
    progressbar = tqdm(executor.map(lambda note: chunk_by_sentence(note), notes), total=len(notes))
    notes = list(progressbar)
    # notes = [section for note in notes for section in note]


df['NOTE'] = notes
df = df.explode('NOTE', ignore_index=True)
notes = df["NOTE"].tolist()


system_prompt = '''
You are a helpful AI Assistant. You will be provided a fragment of a patient's medical note. You must accomplish the following task:

Thinking like a doctor, you are to create a question a doctor might want to know reagrding a patient's medical history.
Create your query question like a doctor would, while keeping the question simple and not to complex.
The answer to this question must be contained in the provided note. 

Format your output as follows:

**Query** 
'''


def llm_1(notes: list[str], system_prompt:str) -> list[str]:
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

def chunked(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

queries = []

for batch in chunked(notes, 8):
    queries.extend(llm_1(batch, system_prompt))

# # Enhanced Formating
# queries = ['Subject ID: ' + str(df['SUBJECT_ID'].iloc[i]) + '\n'  + query[20:-2] for i, query in enumerate(queries)]
# passages = ['Subject ID: ' + str(df['SUBJECT_ID'].iloc[i]) + '\n'  + str(df['NOTE'].iloc[i]) for i in range(len(df))]

queries = [query[13:-2].replace("Query:", "").strip() for query in queries]
passages = [str(df['NOTE'].iloc[i]) for i in range(len(df))]

# df = pd.DataFrame({
#     'QUERY': queries,
#     'PASSAGE': passages
# })

df['QUERY'] = queries
df['PASSAGE'] = notes

df.to_csv("/work/mbouthil/projects/research_project/MEDRAG/processed_data/synq.csv", index=False)