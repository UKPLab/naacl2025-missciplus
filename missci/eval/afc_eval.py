import json
from collections import defaultdict
from typing import Dict, List, Set, Iterable

from sklearn.metrics import classification_report

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.util.fileutil import read_jsonl, write_json, write_jsonl
from missci.util.passage_util import get_gold_fallacy_mapping_dict


def eval_afc_task(prediction_file: str) -> Dict:
    """
    Evaluate an AFC prediction file and store the metrics in the same directory. It expects to contain fields
    * "label_name" for the gold label and
    * "prediction->label" for the predicted label.

    :param prediction_file:
    :return:
    """
    if not prediction_file.endswith('.jsonl'):
        raise ValueError('Invalid filename: ' + prediction_file)

    metrics_file: str = prediction_file.replace('.jsonl', '.metrics.json')
    predictions: List[Dict] = list(read_jsonl(prediction_file))

    gold_labels: List[str] = list(map(lambda x: x['label_name'], predictions))
    predicted_labels: List[str] = list(map(lambda x: x['prediction']['label'], predictions))

    known_labels: Set[str] = {
        'SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'
    }
    if set(gold_labels) - known_labels != set():
        raise ValueError(f'Unknown labels: {list(set(gold_labels) - known_labels)}!')
    if set(predicted_labels) - known_labels != set():
        raise ValueError(f'Unknown prediction: {list(set(predicted_labels) - known_labels)}!')

    metrics: Dict = classification_report(gold_labels, predicted_labels, output_dict=True)
    write_json(metrics, metrics_file, pretty=True)
    return metrics


def to_argument_predictions(raw_prediction_file: str) -> List[Dict]:
    """
    Converts individual predictions based on each passage (of an argument) to unified predictions grouped per argument
    where each argument lists all its passage-level predictions.
    :param raw_prediction_file:
    :return:
    """
    predictions: Iterable[Dict] = read_jsonl(raw_prediction_file)
    arg_to_predictions: Dict[str, Dict] = defaultdict(dict)
    for prediction in predictions:
        prediction = {
            k: prediction[k] for k in prediction.keys() if k not in {'input_ids', 'token_type_ids', 'attention_mask'}
        }
        arg_id: str = prediction['argument_id']
        passage_id: str = prediction['passage_id']
        arg_to_predictions[arg_id][passage_id] = prediction

    arg_entries: List[Dict] = []
    for arg_id in arg_to_predictions.keys():
        arg_entries.append({
            'id': arg_id,
            'passage_predictions': arg_to_predictions[arg_id]
        })
    return arg_entries


def make_four_way_prediction(passage_label_predictions: List[str]) -> str:
    """
    Make a prediction based on the predicted veracity labels of multiple passages of an argument
    :param passage_label_predictions:
    :return:
    """
    valid_labels: Set[str] = {'CONTRADICT', 'SUPPORT', 'NOT_ENOUGH_INFO'}
    if not set(passage_label_predictions) - valid_labels == set():
        raise ValueError(f'Unknown labels: {list(set(passage_label_predictions) - valid_labels)}!')

    passage_label_predictions: Set[str] = set(passage_label_predictions)
    if len(passage_label_predictions) == 1:
        return passage_label_predictions.pop()
    elif len(passage_label_predictions) == 2:
        if 'NOT_ENOUGH_INFO' in passage_label_predictions:
            return (passage_label_predictions - {'NOT_ENOUGH_INFO'}).pop()
        else:
            return 'MIXED'
    else:
        assert len(passage_label_predictions) == 3, passage_label_predictions
        return 'MIXED'


def eval_argument_afc(arg_prediction_file: str, split: str) -> Dict[str, int]:
    """
    Evaluates a file with argument level predictions (must have transformed with "to_argument_predictions()" first.
    :param arg_prediction_file:
    :param split:
    :return:
    """

    # Gold references
    gold_arguments: List[Dict] = MappedDataLoader().load_raw_arguments(split)
    argument_dict = {
        arg['id']: arg for arg in gold_arguments
    }

    # Predictions
    predicted_arguments: List[Dict] = list(read_jsonl(arg_prediction_file))

    # Argument level (4way) predictions
    lbl_counts = defaultdict(int)
    for arg_prediction in predicted_arguments:
        annotated_passages: List[str] = list(argument_dict[arg_prediction['id']]['study']['selected_passages'].keys())
        labels = list(map(
            lambda passage: arg_prediction['passage_predictions'][passage]['prediction']['label'], annotated_passages
        ))
        lbl_counts[make_four_way_prediction(labels)] += 1

    # Avg metrics
    metrics = {
        k: lbl_counts[k] / len(predicted_arguments) for k in lbl_counts.keys()
    }

    for k in lbl_counts.keys():
        metrics[f'count-{k}'] = lbl_counts[k]

    passage_level_metrics = get_passage_level_metrics(argument_dict, predicted_arguments)
    metrics['passage-flagged-p'] = passage_level_metrics['FALLACY']['precision']
    metrics['passage-flagged-r'] = passage_level_metrics['FALLACY']['recall']
    metrics['passage-flagged-f1'] = passage_level_metrics['FALLACY']['f1-score']
    metrics['passage-flagged-support'] = passage_level_metrics['FALLACY']['support']
    metrics['passage-support'] = passage_level_metrics['macro avg']['support']
    for lbl in sorted(list(metrics.keys())):
        print(lbl, '->', metrics[lbl])

    return metrics


def get_passage_level_metrics(gold_argument_dict: Dict[str, Dict], argument_predictions: List[Dict]) -> Dict:

    label2flagged: Dict = {
        'NOT_ENOUGH_INFO': 'NO-FALLACY', 'SUPPORT': 'NO-FALLACY', 'CONTRADICT': 'FALLACY'
    }

    prediction_dict: Dict = defaultdict(dict)
    for prediction in argument_predictions:
        argument_id: str = prediction['id']
        for passage_id in prediction['passage_predictions']:
            predicted_lbl: str = prediction['passage_predictions'][passage_id]['prediction']['label']
            prediction_dict[argument_id][passage_id] = label2flagged[predicted_lbl]

    flagged_passages_gold: List[str] = []
    flagged_passages_pred: List[str] = []

    for argument_id in gold_argument_dict:
        gold_argument: Dict = gold_argument_dict[argument_id]
        predictions: Dict = prediction_dict[argument_id]

        gold_passage_flagging_dict: Dict[str, bool] = get_gold_fallacy_mapping_dict(
            gold_argument, add_empty_context=True
        )

        for passage_id in gold_passage_flagging_dict.keys():
            gold_lbl = 'FALLACY' if gold_passage_flagging_dict[passage_id] else 'NO-FALLACY'
            flagged_passages_gold.append(gold_lbl)
            flagged_passages_pred.append(predictions[passage_id])

    return classification_report(flagged_passages_gold, flagged_passages_pred, output_dict=True)


def evaluate_afc_prediction_file(target_task: str, raw_prediction_file: str, split: str) -> Dict:
    if target_task != 'missci':
        # Direct eval
        metrics: Dict = eval_afc_task(raw_prediction_file)
    else:
        argument_predictions: List[Dict] = to_argument_predictions(raw_prediction_file)
        if not raw_prediction_file.endswith('.jsonl'):
            raise ValueError(f'Invalid file: {raw_prediction_file}')
        argument_prediction_file: str = raw_prediction_file.replace('.jsonl', '.args.jsonl')
        write_jsonl(argument_prediction_file, argument_predictions)
        metrics: Dict = eval_argument_afc(argument_prediction_file, split)

    print(json.dumps(metrics, indent=2))
    return metrics

