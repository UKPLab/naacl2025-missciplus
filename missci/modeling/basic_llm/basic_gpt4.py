import time
from datetime import datetime
from typing import Dict, Optional, List

import openai
from openai.lib.azure import AzureOpenAI

from missci.modeling.basic_llm.llm import LLM
from missci.modeling.prompting import filled_template_to_prompt_gpt
from missci.util.fileutil import read_json


class BasicGPT4(LLM):

    def prompt_with_special_tokens(self, prompt_text: str):
        return self.prompt(filled_template_to_prompt_gpt(prompt_text))

    def specs_string(self) -> str:
        return self.get_specs()["llm"] + "_" + self.get_specs()["gpt_version"]

    def get_specs(self) -> Dict:
        return {
            'llm': 'gpt4',
            'api_version': self.api_version,
            'gpt_version': self.gpt_version
        }

    def __init__(
            self,
            gpt_version: Optional[str] = 'gpt-4',
            api_version: str = '2023-10-01-preview',
            max_new_token_len: int = 1000
          ):

        self.gpt_version: str = gpt_version
        self.api_version: str = api_version
        self.max_new_token_len: int = max_new_token_len
        self.try_again_in: List[int] = [
            10, 30, 30, 30, 30, 60, 60, 60, 60, 60, 90
        ]

        credentials: Dict = read_json('llm-config.json')[gpt_version]
        self.client: AzureOpenAI = AzureOpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            api_version=api_version,
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"]
        )

    def prompt(self, prompt: str) -> Dict:
        current_attempt: int = 0
        while True:
            try:
                messages = [
                    {
                        "role": "user",
                        'content': prompt
                    }]

                completion = self.client.chat.completions.create(
                    model=self.gpt_version, messages=messages, max_tokens=self.max_new_token_len
                )
                output = completion.choices[0].message.content
                usage = {
                    'completion_tokens': completion.usage.completion_tokens,
                    'prompt_tokens': completion.usage.prompt_tokens,
                    'total_tokens': completion.usage.total_tokens,
                    'timestamp': str(datetime.now())
                }
                return {
                    'output': output, 'params': {
                        'usage': usage,
                        'max_new_token_len': self.max_new_token_len,
                        'api_version': self.api_version,
                        'gpt_version': self.gpt_version
                    }
                }
            except openai.RateLimitError as err:
                current_attempt = min((len(self.try_again_in) - 1, current_attempt))
                time.sleep(self.try_again_in[current_attempt])
                current_attempt += 1
