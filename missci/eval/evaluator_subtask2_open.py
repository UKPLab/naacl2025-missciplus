from collections import defaultdict
from os.path import join
from typing import List, Dict, Optional, Set

import numpy as np

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.ranking_metrics import average_precision
from missci.util.directory_util import get_prediction_directory
from missci.util.fileutil import read_jsonl, write_json
from missci.util.passage_util import get_gold_fallacy_mapping_dict, get_passage_mapping_dict_for_fallacies


def passage_wise_p_at_k(ranked_passages: List[str], gold_instance: Dict, k: int) -> float:
    gold_fallacy_mappings: Dict = get_gold_fallacy_mapping_dict(gold_instance)
    gold_passages: Set[str] = {
        passage_key for passage_key in gold_fallacy_mappings if gold_fallacy_mappings[passage_key]
    }

    top_k_passages: List[str] = ranked_passages[:k]
    return len(list(filter(lambda x: x in gold_passages, top_k_passages))) / len(top_k_passages)


def passage_wise_r_at_k(ranked_passages: List[str], gold_instance: Dict, k: Optional[int]) -> float:
    """
    Ignoring fallacies that can be spotted from p0.
    :return:
    """

    gold_fallacies: List[Dict] = gold_instance['argument']['fallacies']

    # these fallacies are found based on p0
    found_fallacy_ids: Set[str] = set(
        [fallacy['id'] for fallacy in gold_fallacies if len(fallacy['fallacy_context'].strip()) == 0]
    )

    passage_to_fallacies = defaultdict(list)
    for gold_fallacy in gold_fallacies:
        for mapping in gold_fallacy['mapping']:
            passage_to_fallacies[mapping['passage']].append(gold_fallacy['id'])

    gold_fallacy_ids: Set[str] = set(map(lambda x: x['id'], gold_fallacies))
    if k is not None:
        ranked_passages = ranked_passages[:k]

    hit_fallacy_ids: Set[str] = set([
        fallacy_id for passage in ranked_passages for fallacy_id in passage_to_fallacies[passage]
    ])
    hit_fallacy_ids |= found_fallacy_ids

    #  our hits must be a true subset
    assert gold_fallacy_ids | hit_fallacy_ids == gold_fallacy_ids

    if len(gold_fallacy_ids) == 0:
        return 1.
    else:
        return len(hit_fallacy_ids) / len(gold_fallacy_ids)


class Subtask2EvaluatorOpen:
    def __init__(
            self,
            prediction_directory: Optional[str] = None,
            data_directory: Optional[str] = None,
            only_evaluate_predicted: bool = False,
            split: str = 'test'
    ):
        self.prediction_directory: str = prediction_directory or get_prediction_directory('subtask2-open')
        self.data_loader: MappedDataLoader = MappedDataLoader(data_directory)
        self.gold_instances: List[Dict] = self.data_loader.load_raw_arguments(split)
        self.only_evaluate_predicted: bool = only_evaluate_predicted

    def evaluate_file(self, file_name: str) -> Dict:
        predictions: List[Dict] = list(read_jsonl(join(self.prediction_directory, file_name)))
        scores: Dict = self.evaluate_instances(predictions)
        score_file_name = file_name.replace('.jsonl', '.json')
        score_file_name = f'evaluation__{score_file_name}'
        write_json(scores, join(self.prediction_directory, score_file_name), pretty=True)
        return scores

    def evaluate_instances(self, predictions: List[Dict]) -> Dict:

        prediction_dict: Dict[str, Dict[str, bool]] = {
            pred['id']: pred for pred in predictions
        }

        if self.only_evaluate_predicted:
            predicted_ids: Set[str] = set(map(lambda x: x['id'], predictions))
            mapped_gold_instances: List[Dict] = list(filter(lambda x: x['id'] in predicted_ids, self.gold_instances))
        else:
            mapped_gold_instances: List[Dict] = self.gold_instances

        scores: Dict[str, List[float]] = defaultdict(list)

        for gold_instance in mapped_gold_instances:
            pred: Dict = prediction_dict[gold_instance['id']]

            for k in [1, 3, 5, 10]:
                scores[f'P@{k}'].append(passage_wise_p_at_k(pred['ranked_passages'], gold_instance, k))

            for k in [0, 1, 3, 5, 10, None]:
                if k is not None:
                    scores[f'R@{k}'].append(passage_wise_r_at_k(pred['ranked_passages'], gold_instance, k))
                else:
                    scores[f'R@All'].append(passage_wise_r_at_k(pred['ranked_passages'], gold_instance, k))

            # mAP
            gold_mapped_dict: Dict[str, bool] = get_passage_mapping_dict_for_fallacies(
                gold_instance, use_full_study=True
            )
            scores['mAP'].append(average_precision(pred['ranked_passages'], gold_mapped_dict))

        agg_scores = {
            key: float(np.mean(scores[key])) for key in scores
        }
        agg_scores['num_arguments'] = len(mapped_gold_instances)
        return agg_scores
