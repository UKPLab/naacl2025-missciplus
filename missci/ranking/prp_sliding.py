import os
import random
from os.path import join, exists
from typing import Dict, List, Optional

from transformers import set_seed

from missci.modeling.basic_llm.llm import LLM
from missci.prompt_generators.pairwise.base_pairwise_template_filler import BasePairwiseTemplateFiller
from missci.util.directory_util import get_raw_prompt_prediction_directory
from missci.util.fileutil import read_jsonl, append_to_write_jsonl
from missci.util.passage_util import get_sorted_passages


class PRPSliding:
    def __init__(
            self,
            template_filler: BasePairwiseTemplateFiller,
            suffix: str,
            llm: LLM,
            num_iterations: int = 3,
            initial_order: str = 'natural-order',
            overwrite: bool = False,
            random_seed: int = 1,
    ):
        set_seed(random_seed)
        self.llm: LLM = llm
        self.template_filler: BasePairwiseTemplateFiller = template_filler
        self.num_iterations: int = num_iterations
        self.initial_order: str = initial_order
        self.overwrite: bool = overwrite
        self.name: str = template_filler.get_name(suffix)
        self.dest_file: str = join(get_raw_prompt_prediction_directory('subtask1'), self.name)

        self.known_predictions: Dict[str, Dict] = dict()

        if exists(self.dest_file):
            if not overwrite:
                self._read_known_predictions()
            os.remove(self.dest_file)

    def rank_passages(self, instance: Dict, use_full_study: bool = False):
        passages: List[Dict] = self._get_initial_passages(instance, use_full_study)

        rankings: List[List[str]] = []
        for i in range(self.num_iterations):
            passages = self._run_backward_iteration(instance, passages, i)
            rankings.append(list(map(lambda passage: passage['passage_id'], passages)))

        result: Dict = {
            'id': instance['id'],
            'ranked_passages': rankings,
            'experiment_data': {
                'llm': self.llm.get_specs(),
                'template': self.template_filler.prompt_template_name,
                'initial_order': self.initial_order
            }
        }
        return result

    def _get_initial_passages(self, instance: Dict, use_full_study: bool) -> List[Dict]:
        if self.initial_order == 'natural-order':
            keys: List[str] = get_sorted_passages(instance=instance, from_full_study=use_full_study)
        elif self.initial_order == 'random':
            keys: List[str] = get_sorted_passages(instance=instance, from_full_study=use_full_study)
            random.shuffle(keys)
        else:
            raise ValueError(f'unknown order: "{self.initial_order}"!')

        return list(map(lambda key: instance['study']['all_passages'][key], keys))

    def _run_backward_iteration(self, instance: Dict, passages: List[Dict], iteration_idx: int) -> List[Dict]:

        # Go from back to front (so that the last item can be transferred to front in the first run)
        # after each iteration <num_iteration> first entries remain untouched.

        pointer = len(passages) - 1  # last index
        while pointer > iteration_idx:
            current_passage = passages[pointer]
            next_passage = passages[pointer - 1]

            if self._move_passage_a_to_front(instance, current_passage, next_passage):
                # swap
                passages[pointer - 1] = current_passage
                passages[pointer] = next_passage

            pointer -= 1

        return passages

    def _move_passage_a_to_front(self, instance: Dict, passage_a: Dict, passage_b: Dict) -> bool:
        prompt_information: Dict = self.template_filler.create_prompt(instance, passage_a, passage_b)
        if prompt_information['data']['pair_id'] in self.known_predictions:
            prediction: Dict = self.known_predictions[prompt_information['data']['pair_id']]
        else:
            prediction: Dict = self.query_llm(prompt_information['prompt'], prompt_information['data'])

        append_to_write_jsonl(self.dest_file, prediction)

        return prediction['move_a_to_front']

    def query_llm(self, prompt: str, data: Dict) -> Dict:
        output: Dict = self.llm.prompt(prompt)
        output['move_a_to_front'] = self.template_filler.move_a_to_front(
            output['output'], default=False
        )
        output['data'] = data
        return output

    def _read_known_predictions(self):
        for entry in read_jsonl(self.dest_file):
            self.known_predictions[entry['data']['pair_id']] = entry

    def get_name(self, num_iterations: int) -> str:
        name: str = self.name.replace('.jsonl', f'.{num_iterations}it.jsonl')
        return name


