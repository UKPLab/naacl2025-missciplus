import os
import random
from os.path import join
from typing import List, Dict

from missci.util.fileutil import read_text


class ConcatContextPromptGenerator:
    def __init__(self, prompt_template_name: str):
        self.prompt_template_name: str = prompt_template_name
        self.template: str = read_text(join('./prompt_templates', prompt_template_name))
        self.p0_prompt_templates: List[str] = [
            f'p{i}-' for i in range(1, 7)
        ]
        self.is_p0_template = self._is_p0_template()

    def make_prompt_for_instance(self, instance: Dict) -> Dict:

        p0: Dict = {
            'text': instance['argument']['accurate_premise_p0']['premise'],
            'passage': 'p0'
        }

        contexts: List[Dict] = [{
            'text': fallacy['fallacy_context'],
            'passage': fallacy['id']
        }
            for fallacy in instance['argument']['fallacies']
            if len(fallacy['fallacy_context'].strip()) > 0
        ]

        if self.is_p0_template:
            random.shuffle(contexts)
            return self._create_p0_template_prompt(instance, p0, contexts)
        else:
            contexts = [p0] + contexts
            random.shuffle(contexts)
            return self._create_concat_all_passage_prompt(instance, contexts)

    def _create_p0_template_prompt(self, instance: Dict, p0: Dict, contexts: List[Dict]) -> Dict:

        assert '@@claim@@' in self.template
        assert '@@p0@@' in self.template
        assert '@@context@@' in self.template

        prompt: str = self.template.replace(
            '@@claim@@', instance['argument']['claim']
        ).replace(
            '@@context@@', '\n\n'.join([c['text'] for c in contexts[1:]])
        ).replace('@@p0@@', p0['text']).replace(
            '@@system_prompt@@', ''
        )
        assert '@@' not in prompt

        return {
            'prompt': prompt
        } | self._add_prompt_data(instance, contexts)

    def _create_concat_all_passage_prompt(self, instance: Dict, contexts: List[Dict]) -> Dict:

        assert '@@claim@@' in self.template
        assert '@@context@@' in self.template

        prompt: str = self.template.replace(
            '@@claim@@', instance['argument']['claim']
        ).replace(
            '@@context@@', '\n\n'.join([c['text'] for c in contexts])
        ).replace('@@p0@@', contexts[0]['text']).replace(
            '@@system_prompt@@', ''
        )
        assert '@@' not in prompt

        return {
            'prompt': prompt
        } | self._add_prompt_data(instance, contexts)

    def _is_p0_template(self) -> bool:
        _, file = os.path.split(self.prompt_template_name)
        for prefix in self.p0_prompt_templates:
            if file.startswith(prefix):
                return True
        return False

    def _add_prompt_data(self, instance: Dict, passages: List[Dict]) -> Dict:
        return {
            'argument_id': instance['id'],
            'claim': instance['argument']['claim'],
            'passages': passages,
            'prompt_template_name': self.prompt_template_name,
            'is_p0_template': self.is_p0_template
        }
