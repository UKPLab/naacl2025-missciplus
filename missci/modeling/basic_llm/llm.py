from typing import Dict


class LLM:
    def prompt(self, prompt: str) -> Dict:
        raise NotImplementedError()

    def get_specs(self) -> Dict:
        raise NotImplementedError()

    def specs_string(self) -> str:
        raise NotImplementedError()

    def prompt_with_special_tokens(self, prompt_text: str):
        raise NotImplementedError()


