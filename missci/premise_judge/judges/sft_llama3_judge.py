from os.path import join

from transformers import AutoTokenizer, AutoModelForCausalLM

from missci.premise_judge.judges.sft_llm_judge import SFTLLMJudge
from missci.util.fileutil import read_json


class SFTLlama3Judge(SFTLLMJudge):

    def _get_model_path(self) -> str:
        cache_dir: str = read_json(self._llm_config_path)[self._llm_name]['directory']
        key2llama = {'70b': '70B-Instruct', '8b': '8B-Instruct'}
        return join(cache_dir, key2llama[self._size])

    def _create_model(self) -> AutoModelForCausalLM:

        return AutoModelForCausalLM.from_pretrained(
            self._get_model_path(),
            quantization_config=self._bnb_config,
            use_cache=False,
            use_flash_attention_2=False,
            device_map="auto",
        )

    def _create_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self._get_model_path())

    def get_response_template(self) -> str:
        return '<|start_header_id|>assistant<|end_header_id|>'
