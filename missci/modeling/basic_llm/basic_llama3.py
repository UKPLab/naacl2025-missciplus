from abc import ABC
from os.path import join
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from missci.modeling.basic_llm.llm import LLM
from missci.modeling.llama import LlamaCaller, get_llama_caller
from missci.modeling.prompting import filled_template_to_prompt_llama3
from missci.util.fileutil import read_json


class BasicLlama3Pipeline(LLM):
    def __init__(self,
                 llama_size: str,
                 run_8bit: bool = True,
                 llm_config_path: str = 'llm-config.json',
                 temperature: float = 0.3,
                 max_new_token_len: int = 3000,
                 ):
        self.max_new_token_len = max_new_token_len
        self.llama_size = llama_size
        self.run_8bit = run_8bit
        self.temperature = temperature
        cache_dir: str = read_json(llm_config_path)['llama3']['directory']
        key2llama = {
            '70b': '70B-Instruct', '8b': '8B-Instruct'
        }
        stored_model_path: str = join(cache_dir, key2llama[llama_size])

        llama3 = AutoModelForCausalLM.from_pretrained(
                stored_model_path,
                device_map="auto",
                load_in_8bit=self.run_8bit
            )

        self.model = pipeline(
            'text-generation',
            #model=stored_model_path,
            model=llama3,
            tokenizer=AutoTokenizer.from_pretrained(stored_model_path),
            model_kwargs={"torch_dtype": torch.bfloat16},
            #device="cuda",
        )

    def prompt(self, prompt_message: str) -> Dict:
        print('Prompt with', prompt_message)
        messages = [
            {"role": "user", "content": prompt_message},
        ]
        prompt = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model(
            prompt,
            max_new_tokens=self.max_new_token_len,
            eos_token_id=terminators,
            temperature=self.temperature
        )

        return {
            'answer': outputs[0]["generated_text"][len(prompt):],
            'output': outputs[0]["generated_text"][len(prompt):]
        }

    def get_specs(self) -> Dict:
        return {
            'llm': 'llama3',
            'size': self.llama_size,
            'temperature': self.temperature,
            'run_8bit': self.run_8bit
        }

    def specs_string(self) -> str:
        s: str = f'llama3_{self.llama_size}_t-{self.temperature}'
        if self.run_8bit:
            s += '_8bit'
        return s

    def prompt_with_special_tokens(self, prompt_text: str):
        return self.prompt(prompt_text)


class BasicLlama3(LLM):

    def prompt_with_special_tokens(self, prompt_text: str):
        return self.prompt(filled_template_to_prompt_llama3(prompt_text))

    def specs_string(self) -> str:
        s: str = f'llama3_{self.llama_size}_t-{self.temperature}'
        if self.run_8bit:
            s += '_8bit'
        return s

    def get_specs(self) -> Dict:
        return {
            'llm': 'llama3',
            'size': self.llama_size,
            'temperature': self.temperature,
            'run_8bit': self.run_8bit

        }

    def __init__(
            self,
            llama_size: str,
            max_prompt_len: int = 5000,
            max_new_token_len: int = 1000,
            temperature: Optional[float] = None,
            run_8bit: bool = True,
            llm_config_path: str = 'llm-config.json',
    ):
        self.llama_size: str = llama_size
        self.temperature = temperature
        self.max_prompt_len = max_prompt_len
        self.max_new_token_len = max_new_token_len
        self.run_8bit = run_8bit or llama_size != '70b'
        key2llama = {
            '70b': '70B-Instruct', '8b': '8B-Instruct'
        }

        cache_dir: str = read_json(llm_config_path)['llama3']['directory']
        stored_model_path: str = join(cache_dir, key2llama[llama_size])
        self.generation_config = GenerationConfig.from_pretrained(stored_model_path)

        if run_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                stored_model_path,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(stored_model_path)

        self.model = model.eval()
        is_cuda = next(model.parameters()).is_cuda
        assert is_cuda, 'no cuda'

        self.tokenizer = AutoTokenizer.from_pretrained(stored_model_path)

    def get_output(self, prompt: str) -> Dict:
        """
        Single batch prompting
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_prompt_len + 1,
            truncation=True,
            padding=False,
            add_special_tokens=False
        ).to(self.model.device)
        if inputs['input_ids'].size()[1] > self.max_new_token_len:
            # Rather do it this way than truncating
            raise ValueError(
                f"Too long input: {inputs['input_ids'].size()} (expected max. {self.max_prompt_len})"
            )

        if self.temperature is not None:
            with torch.inference_mode():
                output_dict = self.model.generate(
                    **inputs,
                    # input_ids=inputs["input_ids"].to("cuda"),
                    max_new_tokens=self.max_new_token_len,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=1,
                    num_return_sequences=1,
                    temperature=self.temperature,
                    do_sample=True,
                    generation_config=self.generation_config
                )
        else:
            with torch.inference_mode():
                output_dict = self.model.generate(
                    **inputs,
                    # input_ids=inputs["input_ids"].to("cuda"),
                    max_new_tokens=self.max_new_token_len,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=1,
                    num_return_sequences=1,
                    generation_config=self.generation_config
                )

        scores = output_dict['scores']
        sequences = output_dict['sequences']

        transition_scores = self.model.compute_transition_scores(
            sequences, scores, normalize_logits=True
        )

        prompt_len: int = inputs["input_ids"].size()[1] + 1  # because of " "
        assert inputs["input_ids"].size()[0] == 1, 'code below is only for single batch'

        transition_scores = transition_scores[:, :prompt_len][0].cpu().numpy()
        print()
        sequences = sequences[:, prompt_len:]
        output_text: str = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        log_probabilities = np.exp(transition_scores)

        return {
            'answer': output_text,
            'transition_scores': list(map(float, transition_scores)),
            'log_probabilities': list(map(float, log_probabilities))
        }

    def prompt(self, prompt: str) -> Dict:
        return {
            'params:': {'llama_size': self.llama_size},
            'output': self.get_output(prompt)['answer']
        }
