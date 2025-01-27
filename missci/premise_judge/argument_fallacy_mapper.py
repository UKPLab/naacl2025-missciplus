from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Set, Tuple

from missci.premise_judge.premise_judge import PremiseJudge
from missci.util.passage_util import get_fallacy_to_passages_mapping


def get_interchangeable_gold_fallacy_dict(gold_argument: Dict) -> Dict[str, List[Dict]]:
    result: Dict[str, List[Dict]] = defaultdict(list)
    fallacy_id_to_gold_passage_keys: Dict[str, List[str]] = get_fallacy_to_passages_mapping(gold_argument)
    for fallacy in gold_argument['argument']['fallacies']:
        for interchangeable_fallacy in fallacy['interchangeable_fallacies']:
            fallacy_class: str = interchangeable_fallacy['class']
            fallacious_premise: str = interchangeable_fallacy['premise']
            interchangeable_fallacy_id: str = interchangeable_fallacy['id']
            result[fallacy_class].append({
                'premise': fallacious_premise,
                'class': fallacy_class,
                'id': interchangeable_fallacy_id,
                'passages': fallacy_id_to_gold_passage_keys[interchangeable_fallacy_id],
                'fallacy_id': fallacy['id']
            })

    return result


def get_predicted_fallacy_dict(predicted: Dict) -> Dict[str, List[Dict]]:
    result: Dict[str, List[Dict]] = defaultdict(list)
    for predicted_fallacy in predicted['all_predictions']['fallacies']:
        fallacy_class: str = predicted_fallacy['normalized_fallacy_class']
        fallacious_premise: str = predicted_fallacy['fallacious_premise']
        passages: str = predicted_fallacy['from']
        prediction_id: str = predicted_fallacy['prediction_id']
        result[fallacy_class].append({
            'premise': fallacious_premise,
            'class': fallacy_class,
            'id': prediction_id,
            'passages': passages,
        })
    return result


class ArgumentFallacyMapper:
    def __init__(self, judge: PremiseJudge):
        self._judge: PremiseJudge = judge

    def map(self, pred: Dict, gold: Dict) -> Dict:
        pred = deepcopy(pred)
        gold_fallacies: Dict[str, List[Dict]] = get_interchangeable_gold_fallacy_dict(gold)
        predicted_fallacies: Dict[str, List[Dict]] = get_predicted_fallacy_dict(pred)

        aligned_argument, aligned_counts = self._make_global_alignment(predicted_fallacies, gold_fallacies)
        pred['alignment'] = aligned_argument
        pred['pred_to_aligned_count'] = aligned_counts
        return pred

    def _make_global_alignment(
            self, predicted_fallacies: Dict[str, List[Dict]], gold_fallacies: Dict[str, List[Dict]]
    ) -> Tuple[Dict, Dict]:
        mapping_dict: Dict = defaultdict(list)
        mapping_counts: Dict = dict()
        for k in predicted_fallacies.keys():
            for pred in predicted_fallacies[k]:
                assert pred['id'] not in mapping_counts
                mapping_counts[pred['id']] = 0

        for fallacy_class in sorted(list(gold_fallacies.keys())):
            gold_fallacies_with_class: List[Dict] = gold_fallacies[fallacy_class]
            pred_fallacies_with_class: List[Dict] = predicted_fallacies[fallacy_class]

            for gold_fallacy in gold_fallacies_with_class:
                for pred_fallacy in pred_fallacies_with_class:
                    assert pred_fallacy['class'] == gold_fallacy['class']
                    premise_gold: str = gold_fallacy['premise']
                    premise_pred: str = pred_fallacy['premise']
                    is_same_reasoning: bool = self._judge.predict_instance(premise_pred, premise_gold)['predicted']
                    if is_same_reasoning:
                        mapping_dict[gold_fallacy['id']].append(pred_fallacy['id'])
                        mapping_counts[pred_fallacy['id']] += 1

        return mapping_dict, mapping_counts

