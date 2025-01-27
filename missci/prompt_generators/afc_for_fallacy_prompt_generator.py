import os.path
from os.path import join
from typing import List, Dict

from missci.util.fileutil import read_text


class AFCFallacyPromptGenerator:

    def __init__(self, prompt_template_name: str):
        self.prompt_template_name: str = prompt_template_name
        self.template: str = read_text(join('./prompt_templates', prompt_template_name))
        self.p0_prompt_templates: List[str] = [
            f'p{i}-' for i in range(1, 7)
        ]
        self.is_p0_template = self._is_p0_template()

    def _is_p0_template(self) -> bool:
        _, file = os.path.split(self.prompt_template_name)
        for prefix in self.p0_prompt_templates:
            if file.startswith(prefix):
                return True
        return False

    def make_prompt_for_instance(self, instance: Dict) -> Dict:
        if self.is_p0_template:
            return self._create_p0_template_prompt(instance)
        else:
            return self._create_concat_all_passage_prompt(instance)

    def _create_concat_all_passage_prompt(self, instance: Dict) -> Dict:

        assert '@@claim@@' in self.template
        assert '@@context@@' in self.template

        prompt: str = self.template.replace(
            '@@claim@@', instance['claim']
        ).replace(
            '@@context@@', instance['evidence']
        ).replace(
            '@@system_prompt@@', ''
        )
        assert '@@' not in prompt

        return {
            'prompt': prompt
        } | self._add_prompt_data(instance)

    def _create_p0_template_prompt(self, instance: Dict) -> Dict:

        assert '@@claim@@' in self.template
        assert '@@p0@@' in self.template
        assert '@@context@@' in self.template

        prompt: str = self.template.replace(
            '@@claim@@', instance['claim']
        ).replace(
            '@@context@@', ''
        ).replace('@@p0@@', instance['evidence']).replace(
            '@@system_prompt@@', ''
        )
        assert '@@' not in prompt

        return {
            'prompt': prompt
        } | self._add_prompt_data(instance)

    def _add_prompt_data(self, instance: Dict) -> Dict:
        return {
            'instance': instance,
            'prompt_template_name': self.prompt_template_name,
            'is_p0_template': self.is_p0_template,
        }
