import json
from collections import defaultdict
from copy import deepcopy
from os.path import join
from typing import Optional, List, Dict, Union, Set, Tuple

import numpy as np

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.missciplus_metrics import compute_claim_level_multi_label_metrics, \
    compute_fallacy_level_alignment_metrics
from missci.util.directory_util import get_prediction_directory
from missci.util.fileutil import read_jsonl, write_json
from missci.util.passage_util import get_passage_mapping_dict_for_fallacies, get_p0_passages, \
    get_context_mapping_dict_for_fallacies


def get_prompt_version_from_file_name(file_name: str):
    if 'st3-passage-wise_' in file_name:
        return 'passage-wise'
    elif 'st3-concat_' in file_name:
        return 'concat-all'
    elif 'concat-ctx__reconstruct-concat' in file_name:
        return 'concat-context'
    elif 'ctx-wise' in file_name:
        return 'context-wise'
    assert False, file_name


def remove_non_gold_passage_predictions(
        instance_prediction: Dict, gold_argument: Dict, prompt_version: str, only_mapped_context: bool = True
) -> Dict:
    passage_2_relevant: Dict[str, bool] = get_passage_mapping_dict_for_fallacies(gold_argument, use_full_study=False)

    fallacy_passage_keys: Set[str] = {p for p in passage_2_relevant.keys() if passage_2_relevant[p]}
    p0_passage_keys = set(map(lambda p: p['passage'], get_p0_passages(
        gold_argument, use_p0_as_backup=True, add_section_title=False
    )))
    used_passage_concatenations: Set[Tuple] = set([
        tuple(fallacy['from']) for fallacy in instance_prediction['prompt_based_predictions']
    ])
    if prompt_version in {'concat-context', 'context-wise'}:
        if only_mapped_context:
            instance_prediction = deepcopy(instance_prediction)
            context_to_relevant = get_context_mapping_dict_for_fallacies(gold_argument)
            fallacy_context_keys: Set[str] = {p for p in context_to_relevant.keys() if context_to_relevant[p]}
            fallacy_context_keys |= {'accurate_premise'}

            # Adjust

            for pred in instance_prediction['prompt_based_predictions']:
                if set(pred['from']) & fallacy_context_keys == set(pred['from']):
                    pass
                else:
                    print('SKIP:', gold_argument['id'], pred['from'])

            instance_prediction['prompt_based_predictions'] = [
                pred for pred in instance_prediction['prompt_based_predictions']
                if set(pred['from']) & fallacy_context_keys == set(pred['from'])
            ]
            instance_prediction['all_predictions']['fallacies'] = [
                fallacy | {'from': pred['from']}
                for pred in instance_prediction['prompt_based_predictions']
                for fallacy in pred['fallacies']
                if set(pred['from']) & fallacy_context_keys == set(pred['from'])
            ]
            kept_pred_ids: Set[str] = {p['prediction_id'] for p in instance_prediction['all_predictions']['fallacies']}
            new_alignment: Dict = dict()
            for key in instance_prediction['alignment']:
                aligned_predictions = [pid for pid in instance_prediction['alignment'][key] if pid in kept_pred_ids]
                if len(aligned_predictions) > 0:
                    new_alignment[key] = aligned_predictions
            instance_prediction['alignment'] = new_alignment

        else:
            # we can keep as is (only have gold)
            return instance_prediction
    elif prompt_version == 'concat-all':
        # every prediction is from same set of fallacies

        assert len(used_passage_concatenations) == 1
        used_passages: Set[str] = {
            passage for concatenation in used_passage_concatenations for passage in concatenation
        }
        # we have used ALL fallacy passages
        assert fallacy_passage_keys & used_passages == fallacy_passage_keys
        assert len(used_passages - fallacy_passage_keys) <= 1
        if len(used_passages - fallacy_passage_keys) == 1:
            passage = next(iter(used_passages - fallacy_passage_keys))
            assert passage in p0_passage_keys or passage == 'accurate_premise'
    elif prompt_version == 'passage-wise':
        instance_prediction = deepcopy(instance_prediction)
        # first get p0 passage
        p0_tuples = [t for t in used_passage_concatenations if len(t) == 1]
        assert len(p0_tuples) == 1, used_passage_concatenations
        p0_key = p0_tuples[0][0]
        assert p0_key in p0_passage_keys or p0_key == 'accurate_premise', p0_key

        # pretend this is a fallacy as well
        fallacy_passage_keys.add(p0_key)
        instance_prediction['prompt_based_predictions'] = [
            pred for pred in instance_prediction['prompt_based_predictions']
            if set(pred['from']) & fallacy_passage_keys == set(pred['from'])
        ]
        instance_prediction['all_predictions']['fallacies'] = [
            fallacy | {'from': pred['from']}
            for pred in instance_prediction['prompt_based_predictions']
            for fallacy in pred['fallacies']
            if set(pred['from']) & fallacy_passage_keys == set(pred['from'])
        ]
        kept_pred_ids: Set[str] = {p['prediction_id'] for p in instance_prediction['all_predictions']['fallacies']}
        new_alignment: Dict = dict()
        for key in instance_prediction['alignment']:
            aligned_predictions = [pid for pid in instance_prediction['alignment'][key] if pid in kept_pred_ids]
            if len(aligned_predictions) > 0:
                new_alignment[key] = aligned_predictions
        instance_prediction['alignment'] = new_alignment
    else:
        raise NotImplementedError(prompt_version)

    return instance_prediction


def get_top_k_predictions(arg_prediction: Dict, k: int, remove_other: bool) -> List[Dict]:
    all_fallacy_predictions: List[Dict] = [
        pred for prompt_pred in arg_prediction['prompt_based_predictions']
        for pred in prompt_pred['fallacies']
        if not remove_other or (pred['normalized_fallacy_class'] not in {'Other', 'None'} and pred['normalized_fallacy_class'] is not None)
    ]

    all_fallacy_predictions = sorted(all_fallacy_predictions, key=lambda x: x['per-prompt-position-rank'])
    # if we have multiple prompts, we may have several predictions with the same rank
    # Here, we prioritize diversity. I.e. give lower (secondary) rank to fallacies of a rarer class.
    fallacy_counts: Dict[str, int] = defaultdict(int)
    for fallacy in all_fallacy_predictions:
        fallacy_cls: str = fallacy['normalized_fallacy_class']
        num_prev: int = fallacy_counts[fallacy_cls]
        fallacy['sort_penalty'] = num_prev
        fallacy_counts[fallacy_cls] += 1

    all_fallacy_predictions = sorted(all_fallacy_predictions, key=lambda x: (x['per-prompt-position-rank'], fallacy['sort_penalty']))
    top_k_predictions: List[Dict] = all_fallacy_predictions[:k]
    return top_k_predictions


def make_prediction_dict(predictions: List[Dict], k: Union[str, int], remove_other: bool = True) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, List]]]:
    predictions = list(map(deepcopy, predictions))
    prediction_dict: Dict[str, List[Dict]] = dict()

    if k == 'all':
        for arg_pred in predictions:
            prediction_dict[arg_pred['argument']] = []
            for prompt_based_pred in arg_pred['prompt_based_predictions']:
                for fallacy in prompt_based_pred['fallacies']:
                    if not remove_other or (fallacy['normalized_fallacy_class'] not in {'Other', 'None'} and fallacy['normalized_fallacy_class'] is not None):
                        prediction_dict[arg_pred['argument']].append(fallacy)
    else:
        for arg_pred in predictions:
            prediction_dict[arg_pred['argument']] = get_top_k_predictions(arg_pred, k, remove_other=remove_other)

    if remove_other:
        for arg in prediction_dict:
            for f in prediction_dict[arg]:
                assert f['normalized_fallacy_class'] not in {'Other', 'None'} and f['normalized_fallacy_class'] is not None, f['normalized_fallacy_class']

    alignment_dict: Dict[str, Dict[str, List]] = dict()
    for pred in predictions:
        kept_prediction_ids: Set[str] = set([p['prediction_id'] for p in prediction_dict[pred['argument']]])
        alignment = pred['alignment']
        keys = list(alignment.keys())
        for key in keys:
            alignment[key] = [pid for pid in alignment[key] if pid in kept_prediction_ids]
        alignment_dict[pred['argument']] = alignment

    return prediction_dict, alignment_dict


def get_stats(predictions: List[Dict]) -> Dict:
    num_prompts: int = sum([len(p['prompt_based_predictions']) for p in predictions])

    used_passages: Dict[List[str]] = dict()

    num_arg_level_predictions: List[int] = []
    num_prompt_level_predictions: List[int] = []
    for pred in predictions:

        current_used_passages = set()
        num_pred_per_arg: List[int] = []
        for prompt_pred in pred['prompt_based_predictions']:
            num_pred_per_arg.append(len(prompt_pred['fallacies']))
            current_used_passages |= {
                p
                for p in prompt_pred['from']
            }
        used_passages[pred['argument']] = list(current_used_passages)

        num_arg_level_predictions.append(sum(num_pred_per_arg))
        num_prompt_level_predictions.extend(num_pred_per_arg)

    num_parsed_results: int = sum([
        len(list(filter(lambda x: x['is_parsed'], p['prompt_based_predictions']))) for p in predictions
    ])
    return {
            'arg_to_passages': used_passages,
            'arguments': len(predictions),
            'prompts': num_prompts,
            'parsed_absolute': num_parsed_results,
            'parsed_relative': num_parsed_results / num_prompts,
            'predictions_per_prompt': {
                'mean': float(np.mean(num_prompt_level_predictions)),
                'median': float(np.median(num_prompt_level_predictions)),
                'std': float(np.std(num_prompt_level_predictions)),
            },
            'predictions_per_arg': {
                'mean': float(np.mean(num_arg_level_predictions)),
                'median': float(np.median(num_arg_level_predictions)),
                'std': float(np.std(num_arg_level_predictions)),
            },
            'total_predictions': sum(num_arg_level_predictions)
        }


class GenClassifyEvaluatorMissciPlus:
    def __init__(
            self,
            split: str,
            prediction_directory: Optional[str] = None
    ):
        self.split: str = split
        self.prediction_directory: str = prediction_directory or get_prediction_directory('argument-reconstruction')
        self.remove_unk_fallacies: bool = True
        self.gold_instance_dict: Dict = {
            argument['id']: argument
            for argument in MappedDataLoader().load_raw_arguments(split=split)
        }

    def load_predictions(self, file_name: str, use_all_passages: bool, use_mapped_only: bool) -> List[Dict]:
        predictions: List[Dict] = list(read_jsonl(join(self.prediction_directory, file_name)))
        if use_all_passages:
            return predictions
        else:
            prompt_version: str = get_prompt_version_from_file_name(file_name)
            predictions = [
                remove_non_gold_passage_predictions(
                    p, self.gold_instance_dict[p['argument']], prompt_version, only_mapped_context=use_mapped_only
                )
                for p in predictions
            ]
            return predictions

    def make_prediction_dict(self, file_name: str, use_all_passages: bool = False, k: Union[str, int] = 5, mapped_only: bool = False) -> Dict:
        predictions: List[Dict] = self.load_predictions(file_name, use_all_passages, use_mapped_only=mapped_only)
        prediction_dict, alignment_dict = make_prediction_dict(predictions, k)
        return {
            'predictions': prediction_dict,
            'alignment': alignment_dict
        }

    def evaluate_file(self, file_name: str, use_all_passages: bool, use_only_mapped_context: bool) -> Dict:
        predictions: List[Dict] = self.load_predictions(file_name, use_all_passages, use_only_mapped_context)
        scores: Dict = self.evaluate_instances(predictions)
        score_file_name = file_name.replace('.jsonl', '.json')
        if use_all_passages:
            score_file_name = score_file_name.replace('.json', '.all.json')
        else:
            score_file_name = score_file_name.replace('.json', '.gold.json')

        if use_only_mapped_context:
            score_file_name = score_file_name.replace('.json', '.mapped-context.json')

        score_file_name = f'evaluation__{score_file_name}'
        print(json.dumps(scores, indent=2))
        write_json(scores, join(self.prediction_directory, score_file_name), pretty=True)
        return scores

    def evaluate_instances(self, predictions: List[Dict]) -> Dict:
        result = {
            'stats': get_stats(predictions),
            'all': self.evaluate_instances_top_k(predictions, 'all'),
            'top1': self.evaluate_instances_top_k(predictions, 1),
            'top3': self.evaluate_instances_top_k(predictions, 3),
            'top5': self.evaluate_instances_top_k(predictions, 5),
            'top10': self.evaluate_instances_top_k(predictions, 10)
        }
        return result

    def evaluate_instances_top_k(self, predictions: List[Dict], k: Union[str, int]) -> Dict:
        # Stats
        prediction_dict, alignment_dict = make_prediction_dict(predictions, k)

        if k != 'all':
            for key in prediction_dict:
                assert len(prediction_dict[key]) <= k, f'expect: {k}, found:{len(prediction_dict[key])}'
        claim_level_multi_label: Dict = compute_claim_level_multi_label_metrics(
            prediction_dict, self.gold_instance_dict
        )
        fallacy_level_alignment: Dict = compute_fallacy_level_alignment_metrics(
            prediction_dict, alignment_dict, self.gold_instance_dict
        )
        fallacy_level_alignment_no_premise: Dict = compute_fallacy_level_alignment_metrics(
            prediction_dict, None, self.gold_instance_dict, use_premise_alignment=False
        )

        result: Dict = {
            'claim_level_multi_label': claim_level_multi_label,
            'fallacy_level': fallacy_level_alignment,
            'fallacy_level_no_premise': fallacy_level_alignment_no_premise
        }
        # print(json.dumps(result, indent=2))
        return result








