import os.path
import random
from os.path import join
from typing import List, Dict

from missci.util.fileutil import read_text
from missci.util.passage_util import get_p0_passages, get_sorted_passages, get_passage_mapping_dict_for_fallacies, \
    get_passage_text


def sort_passages(passages: List[Dict]) -> List[Dict]:
    return sorted(passages, key=lambda x: int(x['passage'].split('-')[-1]))


class ConcatPassagesPromptGenerator:

    def __init__(self, prompt_template_name: str, passage_order: str, use_all_passages: bool, add_section_title: bool):
        passage_order = passage_order.strip()
        self.prompt_template_name: str = prompt_template_name
        self.template: str = read_text(join('./prompt_templates', prompt_template_name))
        self.p0_prompt_templates: List[str] = [
            f'p{i}-' for i in range(1, 7)
        ]
        self.is_p0_template = self._is_p0_template()
        self.passage_order: str = passage_order
        self.use_all_passages: bool = use_all_passages
        self.add_section_title: bool = add_section_title
        print('#' + passage_order + '#')
        print("passage_order == 'p0-ordered'", passage_order == 'p0-ordered')
        if self.is_p0_template and passage_order not in {'p0-random', 'p0-ordered'}:
            raise ValueError(
                f"must provide on of {['p0-random', 'p0-ordered']} for the template: {prompt_template_name}! (found: {passage_order})"
            )

    def _is_p0_template(self) -> bool:
        _, file = os.path.split(self.prompt_template_name)
        for prefix in self.p0_prompt_templates:
            if file.startswith(prefix):
                return True
        return False

    def make_prompt_for_instance(self, instance: Dict) -> Dict:
        p0_passages: List[Dict] = get_p0_passages(
            instance, add_section_title=self.add_section_title, use_p0_as_backup=True
        )
        random.shuffle(p0_passages)
        passage_keys: List[str] = list(get_sorted_passages(instance, from_full_study=False))
        passage_2_relevant: Dict[str, bool] = get_passage_mapping_dict_for_fallacies(instance, use_full_study=False)

        if not self.use_all_passages:
            passage_keys = [k for k in passage_keys if passage_2_relevant[k]]

        passages: List[Dict] = self._prepare_passages(p0_passages, passage_keys, instance)
        if self.is_p0_template:
            return self._create_p0_template_prompt(instance, passages)
        else:
            return self._create_concat_all_passage_prompt(instance, passages)

    def _create_concat_all_passage_prompt(self, instance: Dict, passages: List[Dict]) -> Dict:

        assert '@@claim@@' in self.template
        assert '@@context@@' in self.template

        prompt: str = self.template.replace(
            '@@claim@@', instance['argument']['claim']
        ).replace(
            '@@context@@', '\n\n'.join([p['text'] for p in passages])
        ).replace('@@p0@@', passages[0]['text']).replace(
            '@@system_prompt@@', ''
        )
        assert '@@' not in prompt

        return {
            'prompt': prompt
        } | self._add_prompt_data(instance, passages)

    def _create_p0_template_prompt(self, instance: Dict, passages: List[Dict]) -> Dict:

        assert '@@claim@@' in self.template
        assert '@@p0@@' in self.template
        assert '@@context@@' in self.template

        prompt: str = self.template.replace(
            '@@claim@@', instance['argument']['claim']
        ).replace(
            '@@context@@', '\n\n'.join([p['text'] for p in passages[1:]])
        ).replace('@@p0@@', passages[0]['text']).replace(
            '@@system_prompt@@', ''
        )
        assert '@@' not in prompt

        return {
            'prompt': prompt
        } | self._add_prompt_data(instance, passages)

    def _prepare_passages(self, p0_passages: List[Dict], passage_keys: List[str], instance: Dict) -> List[Dict]:
        p0_passage: Dict = p0_passages[0]
        passage_keys = filter(lambda key: p0_passage['passage'] != key, passage_keys)
        study: Dict = instance['study']['selected_passages']
        other_passages: List[Dict] = [
            {
                'text': get_passage_text(study[passage_key], add_section_title=self.add_section_title),
                'passage': passage_key
            } for passage_key in passage_keys
        ]

        return self._reorder_passages(p0_passage, other_passages)

    def _reorder_passages(self, p0_passage: Dict, other_passages: List[Dict]) -> List[Dict]:
        if self.passage_order == 'ordered':
            if p0_passage['passage'] is not None:
                return sort_passages(other_passages + [p0_passage])
            else:
                return [p0_passage] + sort_passages(other_passages)
        elif self.passage_order == 'p0-ordered':
            return [p0_passage] + sort_passages(other_passages)
        elif self.passage_order == 'random':
            passages: List[Dict] = other_passages + [p0_passage]
            random.shuffle(passages)
            return passages
        elif self.passage_order == 'p0-random':
            random.shuffle(other_passages)
            return [p0_passage] + other_passages
        else:
            raise NotImplementedError(self.passage_order)

    def _add_prompt_data(self, instance: Dict, passages: List[Dict]) -> Dict:
        return {
            'argument_id': instance['id'],
            'claim': instance['argument']['claim'],
            'passages': passages,
            'prompt_template_name': self.prompt_template_name,
            'passage_order': self.passage_order,
            'is_p0_template': self.is_p0_template,
            'use_all_passages': self.use_all_passages,
            'add_section_title': self.add_section_title
        }


