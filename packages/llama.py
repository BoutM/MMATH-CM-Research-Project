import os
import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, logging, AutoModelForCausalLM
logging.set_verbosity_error()


### Loading LM Model ###
load_dotenv('/work/mbouthil/MMATH-CM-Research-Project/token.env')
hf_token = os.getenv('HUGGINGFACE_TOKEN')
model_name = "meta-llama/Llama-3.1-8B-Instruct"


class Llama_LM:
    def __init__(self,
                 hf_token: str = hf_token,
                 model_name: str = model_name):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt(self,
               messages,
               padding: bool = True,
               truncation: bool = True,
               max_tokens: int = 100,
               temp: float = 0.1,
               top_p: float = 0.9,
               da_wrap: bool = True) -> list[str]:

        # CORRECT device resolution (works with sharded models)
        device = next(self.model.parameters()).device

        prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                do_sample=True
            )

        responses = []

        for i in range(len(messages)):

            # SAFE decoding boundary (handles padding correctly)
            input_len = inputs["attention_mask"][i].sum()
            gen_tokens = outputs[i][input_len:]

            decoded = self.tokenizer.decode(
                gen_tokens,
                skip_special_tokens=True
            )

            responses.append(decoded)

        if da_wrap:
            cleaned = []
            for r in responses:
                match = re.findall(r'\*\*([^*]+)\*\*', r)
                cleaned.append(match[0] if match else r)
            responses = cleaned

        return responses