from abc import ABC
from typing import Dict, Optional

from missci.modeling.basic_llm.llm import LLM
from missci.modeling.llama import LlamaCaller, get_llama_caller
from missci.modeling.prompting import filled_template_to_prompt_llama


class BasicLlama2(LLM):

    def prompt_with_special_tokens(self, prompt_text: str):
        return self.prompt(filled_template_to_prompt_llama(prompt_text))

    def specs_string(self) -> str:
        s: str = f'llama2_{self.llama_size}_t-{self.temperature}'
        if self.run_8bit:
            s += '_8bit'
        return s

    def get_specs(self) -> Dict:
        return {
            'llm': 'llama2',
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
    ):
        self.llama_size: str = llama_size
        self.temperature = temperature
        self.run_8bit = run_8bit or llama_size != '70b'
        key2llama = {
            '70b': '70B-Chat', '13b': '13B-Chat', '7b': '7B-Chat'
        }
        self.llama2: LlamaCaller = get_llama_caller(
            model_variant=key2llama[llama_size],
            max_prompt_input_size=max_prompt_len,
            max_new_tokens=max_new_token_len,
            temperature=temperature,
            load_in_4bit=not self.run_8bit,
            load_in_8bit=self.run_8bit
        )

    def prompt(self, prompt: str) -> Dict:
        return {
            'params:': {'llama_size': self.llama_size},
            'output': self.llama2.get_output(prompt)['answer']
        }
