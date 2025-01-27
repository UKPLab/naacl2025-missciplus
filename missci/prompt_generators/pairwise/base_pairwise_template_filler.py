from os.path import join
from typing import Dict, Iterable, Tuple

from missci.modeling.prompting import filled_template_to_prompt_gpt, filled_template_to_prompt_llama, to_prompt_for_llm
from missci.util.fileutil import read_text


class BasePairwiseTemplateFiller:

    def __init__(
            self,
            prompt_template_name: str,
            llm_architecture: str,
            convert_prompt_format: bool = True,
            prompt_template_dir: str = './prompt_templates',
    ):
        self.prompt_template_name: str = prompt_template_name
        self.llm_architecture: str = llm_architecture
        self.prompt_template: str = read_text(join(prompt_template_dir, prompt_template_name))
        self.convert_prompt_format: bool = convert_prompt_format

    def create_prompt(self, instance: Dict, passage_a: Dict, passage_b: Dict) -> Dict:

        filled_template: str = self._fill_template(instance, passage_a, passage_b)
        if self.convert_prompt_format:
            filled_template: str = to_prompt_for_llm(self.llm_architecture, filled_template)

        return {
            'data': self.get_pair_data(instance, passage_a, passage_b),
            'prompt': filled_template
        }

    def move_a_to_front(self, response: str, default: bool) -> bool:
        raise NotImplementedError()

    def get_pair_data(self, instance: Dict, passage_a: Dict, passage_b: Dict) -> Dict:
        return {
            'argument_id': instance['id'],
            'passage_a': passage_a['passage_id'],
            'passage_b': passage_b['passage_id'],
            'pair_id': f"{instance['id']}__{passage_a['passage_id']}__{passage_b['passage_id']}"
        }

    def _fill_template(self, instance: Dict, passage_a: Dict, passage_b: Dict) -> str:
        raise NotImplementedError()

    def get_name(self, suffix: str):
        prompt_name: str = self.prompt_template_name.replace('\\', '--').replace('/', '--').replace('.txt', '')
        return f'missci__{prompt_name}__{self.llm_architecture}.{suffix}.jsonl'
