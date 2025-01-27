from collections import defaultdict
from os.path import join
from typing import Dict, Optional, List, Set

import numpy as np

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.ranking_metrics import precision_at_k, reciprocal_rank, has_positive_at
from missci.util.directory_util import get_prediction_directory
from missci.util.fileutil import read_jsonl, write_json
from missci.util.passage_util import has_mapped_p0, get_gold_passages


class Subtask1Evaluator:
    def __init__(
            self,
            prediction_directory: Optional[str] = None,
            data_directory: Optional[str] = None,
            only_evaluate_predicted: bool = False,
            use_full_study: bool = False,
            split: str = 'test'
    ):
        self.prediction_directory: str = prediction_directory or get_prediction_directory('subtask1')
        self.data_loader: MappedDataLoader = MappedDataLoader(data_directory)
        self.gold_instances: List[Dict] = self.data_loader.load_raw_arguments(split)
        self.only_evaluate_predicted: bool = only_evaluate_predicted
        self.use_full_study: bool = use_full_study

    def evaluate_file(self, file_name: str, allow_passage_mismatch: bool = False) -> Optional[Dict]:
        predictions: List[Dict] = list(read_jsonl(join(self.prediction_directory, file_name)))
        scores: Optional[Dict] = self.evaluate_instances(
            predictions, file=file_name, allow_passage_mismatch=allow_passage_mismatch
        )
        if scores is None:
            return None

        if 'experiment_data' in predictions[0]:
            scores['experiment_data'] = predictions[0]['experiment_data']

        score_file_name = file_name.replace('.jsonl', '.json')
        score_file_name = f'evaluation__{score_file_name}'
        write_json(scores, join(self.prediction_directory, score_file_name), pretty=True)
        return scores

    def evaluate_instances(
            self, predictions: List[Dict], file: Optional[str] = None, allow_passage_mismatch: bool = False
    ) -> Optional[Dict]:
        # Only evaluate over instances for which we have a mapping
        mapped_gold_instances: List[Dict] = list(filter(has_mapped_p0, self.gold_instances))

        prediction_dict: Dict[str, List[str]] = {
            pred['id']: pred['ranked_passages'] for pred in predictions
        }

        if self.only_evaluate_predicted:
            predicted_ids: Set[str] = set(map(lambda x: x['id'], predictions))
            mapped_gold_instances: List[Dict] = list(filter(lambda x: x['id'] in predicted_ids, mapped_gold_instances))

        all_p_at_1: List[float] = []
        all_mrr: List[float] = []

        has_positives_dict: Dict[str, List] = defaultdict(list)

        # Uses the gold instance as basis (i.e. for each gold instance a prediction is expected)
        # Unless "only_evaluate_predicted" set to True
        for gold_instance in mapped_gold_instances:
            if gold_instance['id'] not in prediction_dict:
                print(f'Missing prediction (file:{file})')
                return None
            gold_passages: List[str] = get_gold_passages(gold_instance['argument']['accurate_premise_p0']['mapping'])
            ranked_passages: List[str] = prediction_dict[gold_instance['id']]

            if self.use_full_study:
                # Make sure we have predictions for all
                all_passage_ids: Set[str] = set(gold_instance['study']['all_passages'].keys())
                if set(ranked_passages) != set(all_passage_ids):
                    raise NotImplementedError(f'Passage Mismatch: {set(ranked_passages)} != {set(all_passage_ids)}!')
            else:
                mapped_passage_ids: Set[str] = set(gold_instance['study']['selected_passages'].keys())
                ranked_passages = list(filter(lambda x: x in mapped_passage_ids, ranked_passages))
                if set(ranked_passages) != set(mapped_passage_ids) and not allow_passage_mismatch:
                    raise NotImplementedError(f'Passage Mismatch: {set(ranked_passages)} != {set(mapped_passage_ids)}!')

            all_p_at_1.append(precision_at_k(gold_passages, ranked_passages, k=1))
            all_mrr.append(reciprocal_rank(gold_passages, ranked_passages))

            for k in [3, 5, 10]:
                has_positives_dict[f'has-positives-{k}'].append(has_positive_at(gold_passages, ranked_passages, k=k))

        scores: Dict = {
            'instances': len(mapped_gold_instances),
            'P@1': float(np.mean(all_p_at_1)),
            'MRR': float(np.mean(all_mrr))
        }

        if 'unparsed_passages' in predictions[0]:
            num_all_passages: int = 0
            num_unparsed_passages: int = 0

            for prediction in predictions:
                num_all_passages += len(prediction['ranked_passages']) + len(prediction['unparsed_passages'])
                num_unparsed_passages += len(prediction['unparsed_passages'])
            scores['parsed-passages'] = (num_all_passages - num_unparsed_passages) / num_all_passages

        for key in has_positives_dict:
            scores[key] = float(np.mean(has_positives_dict[key]))

        return scores
