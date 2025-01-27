from collections import defaultdict
from typing import Dict, List, Set, Iterable, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from missci.output_parser.fallacy_mapper import FALLACY_DICT
from missci.premise_judge.argument_fallacy_mapper import get_interchangeable_gold_fallacy_dict
from missci.util.passage_util import get_fallacies_from_argument


def compute_claim_level_multi_label_metrics(predictions: Dict[str, List[Dict]], gold_instances: Dict):
    multi_label_gold: List[List[str]] = []
    multi_label_pred: List[List[str]] = []

    for argument_id in gold_instances.keys():
        gold_arg: Dict = gold_instances[argument_id]
        pred: List[Dict] = predictions[argument_id]
        predicted_fallacy_classes: Iterable[str] = map(lambda f: f['normalized_fallacy_class'], pred)

        current_pred_fallacies: List[str] = list(set(predicted_fallacy_classes))
        current_gold_fallacies: List[str] = list(set(map(lambda f: f['class'], get_fallacies_from_argument(gold_arg))))

        multi_label_gold.append(current_gold_fallacies)
        multi_label_pred.append(current_pred_fallacies)

    # Compute micro/macro F1
    classes = sorted(list(FALLACY_DICT.keys()))
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])

    encoded_gold = mlb.transform(multi_label_gold)
    encoded_pred = mlb.transform(multi_label_pred)

    fallacy_level_f1: np.ndarray = f1_score(encoded_gold, encoded_pred, zero_division=0, average=None)
    fallacy_level_precision: np.ndarray = precision_score(encoded_gold, encoded_pred, zero_division=0, average=None)
    fallacy_level_recall: np.ndarray = recall_score(encoded_gold, encoded_pred, zero_division=0, average=None)

    metrics: Dict = {
        'f1_micro': float(f1_score(encoded_gold, encoded_pred, average="micro", zero_division=0)),
        'f1_macro': float(f1_score(encoded_gold, encoded_pred, average="macro", zero_division=0)),
        'f1_sample-averaged': float(f1_score(encoded_gold, encoded_pred, average="samples", zero_division=0)),
        'fallacy_wise': {
            fallacy: {
                'precision': float(fallacy_level_precision[i]),
                'recall': float(fallacy_level_recall[i]),
                'f1': float(fallacy_level_f1[i])
            } for i, fallacy in enumerate(mlb.classes_)
        }
    }
    return metrics


def get_all_interchangeable_fallacy_ids(instance: Dict) -> Set[str]:
    return set([
        interchangeable_fallacy['id']
        for fallacy in instance['argument']['fallacies']
        for interchangeable_fallacy in fallacy['interchangeable_fallacies']
    ])


def align_without_premise(predictions: List[Dict], gold_instance: Dict) -> Dict[str, List[str]]:
    gold_dict_fallacies: Dict[str, List[Dict]] = get_interchangeable_gold_fallacy_dict(gold_instance)
    alignment: Dict[str, List[str]] = dict()
    for gold_fallacy_class in gold_dict_fallacies.keys():
        mapped_prediction_ids_to_fallacy = [
            m['prediction_id'] for m in predictions if
            m['normalized_fallacy_class'] == gold_fallacy_class
        ]
        if len(mapped_prediction_ids_to_fallacy) > 0:
            for gold_interchangeable_fallacy in gold_dict_fallacies[gold_fallacy_class]:
                assert gold_interchangeable_fallacy['id'] not in alignment
                alignment[gold_interchangeable_fallacy['id']] = mapped_prediction_ids_to_fallacy
    return alignment


def get_all_aligned_interchangeable_fallacy_ids(
        alignment: Dict[str, List] = None
) -> Set[str]:

    return {
        int_fid for int_fid in alignment.keys()
        if len(alignment[int_fid]) > 0
    }


def correct_premise_alignment(prediction: Dict, gold_arg: Dict):
    old_alignment: Dict = prediction['alignment']

    # no alignment or after bugfix
    if len(old_alignment.keys()) == 0 or len(list(old_alignment.keys())[0].split(':')) == 3:
        return

    fallacy_dict: Dict[str, Dict] = {
        f['id']: f for f in gold_arg['argument']['fallacies']
    }

    prediction_id_to_cls: Dict[str, str] = {
        p['prediction_id']: p['normalized_fallacy_class']
        for p in prediction['all_predictions']['fallacies']
    }

    def get_interchangeable_fallacy_id(interchangeable_fallacies: List[Dict], fallacy_class: str):
        for f in interchangeable_fallacies:
            if f['class'] == fallacy_class:
                return f['id']
        raise ValueError(f'NOT FOUND: {fallacy_class}')

    new_alignment = defaultdict(list)
    for fallacy_id in old_alignment:
        fallacy = fallacy_dict[fallacy_id]
        for prediction_id in old_alignment[fallacy_id]:
            predicted_class = prediction_id_to_cls[prediction_id]
            interchangeable_fallacy_id = get_interchangeable_fallacy_id(fallacy['interchangeable_fallacies'], predicted_class)
            assert prediction_id not in new_alignment[interchangeable_fallacy_id]
            new_alignment[interchangeable_fallacy_id].append(prediction_id)

    prediction['alignment'] = new_alignment


def compute_fallacy_level_alignment_metrics(
        predictions: Dict[str, List[Dict]], alignments: Optional[Dict[str, Dict[str, List]]],
        gold_instances: Dict, use_premise_alignment: bool = True
) -> Dict:

    # For interchangeable fallacy level p/r
    all_num_predicted_interchangeable_fallacies: List[int] = []
    all_num_gold_interchangeable_fallacies: List[int] = []
    all_num_predicted_mapped_interchangeable_fallacies: List[int] = []

    # For necessary fallacy level
    all_num_necessary_fallacies_gold: List[int] = []
    all_num_necessary_fallacies_mapped: List[int] = []

    for argument_id in gold_instances.keys():
        gold_arg: Dict = gold_instances[argument_id]
        predicted_fallacies: List[Dict] = predictions[argument_id]

        # Extract all fallacy ids
        all_gold_interchangeable_fallacies: Set[str] = get_all_interchangeable_fallacy_ids(gold_arg)

        if use_premise_alignment:
            all_aligned_interchangeable_fallacies: Set[str] = get_all_aligned_interchangeable_fallacy_ids(
                alignments[argument_id]
            )
        else:
            all_aligned_interchangeable_fallacies: Set[str] = get_all_aligned_interchangeable_fallacy_ids(
                align_without_premise(predictions[argument_id], gold_arg)
            )

        all_aligned_necessary_fallacies: Set[str] = set(
            map(lambda f_id: ':'.join(f_id.split(':')[:-1]), all_aligned_interchangeable_fallacies)
        )
        all_necessary_fallacy_ids: Set[str] = set(map(lambda f: f['id'], gold_arg['argument']['fallacies']))

        # Update numbers
        all_num_predicted_interchangeable_fallacies.append(len(predicted_fallacies))
        all_num_gold_interchangeable_fallacies.append(len(all_gold_interchangeable_fallacies))
        all_num_predicted_mapped_interchangeable_fallacies.append(len(all_aligned_interchangeable_fallacies))

        all_num_necessary_fallacies_gold.append(len(all_necessary_fallacy_ids))
        all_num_necessary_fallacies_mapped.append(len(all_necessary_fallacy_ids & all_aligned_necessary_fallacies))

    assert len(all_num_predicted_interchangeable_fallacies) == len(gold_instances.keys())
    assert len(all_num_gold_interchangeable_fallacies) == len(gold_instances.keys())
    assert len(all_num_predicted_mapped_interchangeable_fallacies) == len(gold_instances.keys())

    assert len(all_num_necessary_fallacies_gold) == len(gold_instances.keys())
    assert len(all_num_necessary_fallacies_mapped) == len(gold_instances.keys())

    return {

        'interchangeable_fallacy_metrics': {
            'total_predicted': sum(all_num_predicted_interchangeable_fallacies),
            'total_gold': sum(all_num_gold_interchangeable_fallacies),
            'total_mapped': sum(all_num_predicted_mapped_interchangeable_fallacies),
            'recall': sum(all_num_predicted_mapped_interchangeable_fallacies) / sum(all_num_gold_interchangeable_fallacies),
            'precision': sum(all_num_predicted_mapped_interchangeable_fallacies) / sum(all_num_predicted_interchangeable_fallacies),
        },
        'necessary_fallacy_metrics': {
            'total_gold': sum(all_num_necessary_fallacies_gold),
            'total_mapped': sum(all_num_necessary_fallacies_mapped),
            'recall_mapped': sum(all_num_necessary_fallacies_mapped) / sum(all_num_necessary_fallacies_gold),
            'arg@1': len([m for m in all_num_necessary_fallacies_mapped if m > 0]) / len(all_num_necessary_fallacies_gold)
        }
    }
