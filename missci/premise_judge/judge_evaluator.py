import os.path
from os.path import join
from typing import Dict, List
from sklearn.metrics import classification_report

from missci.util.fileutil import write_json


def stringify(lbl: bool) -> str:
    return 'match' if lbl else 'no-match'


class JudgeEvaluator:
    def __init__(self):
        self.predictions: List[str] = []
        self.gold_labels: List[str] = []
        self.fold_metrics: List = []

    def add_fold_predictions(self, predicted_labels: List[bool], gold_labels: List[bool], log: bool = True):
        assert len(predicted_labels) == len(gold_labels)
        self.predictions.extend(list(map(stringify, predicted_labels)))
        self.gold_labels.extend(list(map(stringify, gold_labels)))
        self.fold_metrics.append(
            classification_report(
                list(map(stringify, gold_labels)), list(map(stringify, predicted_labels)),
                output_dict=True, zero_division=0
            )
        )
        if log:
            print(classification_report(
                list(map(stringify, gold_labels)), list(map(stringify, predicted_labels)), zero_division=0
            ))

    def evaluate(self) -> Dict:
        print(classification_report(self.gold_labels, self.predictions, zero_division=0, digits=3))
        result: Dict = classification_report(self.gold_labels, self.predictions, output_dict=True, zero_division=0)
        return result





