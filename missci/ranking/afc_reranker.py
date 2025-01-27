from typing import List, Dict, Callable

import numpy as np

from missci.modeling.afc_inference import get_id2label
from scipy.stats import entropy


class AFCReranker:

    def __init__(self, agg_strategy: str, task_name: str):
        id2label: Dict[int, str] = get_id2label(task_name, is_path=False)
        self.label2id: Dict[str, int] = {id2label[k]: k for k in id2label.keys()}

        if agg_strategy == 's+n':
            self.score_fn: Callable[[List[float]], float] = self._supp_nei
            self.suffix: str = 'support-nei'
        elif agg_strategy == 's':
            self.score_fn: Callable[[List[float]], float] = self._supp
            self.suffix: str = 'support'
        elif agg_strategy == 'r':
            self.score_fn: Callable[[List[float]], float] = self._refute
            self.suffix: str = 'refute'
        elif agg_strategy == 's+r':
            self.score_fn: Callable[[List[float]], float] = self._supp_refute
            self.suffix: str = 'support-refute'
        elif agg_strategy == 'entropy':
            self.score_fn: Callable[[List[float]], float] = self._entropy
            self.suffix: str = 'entropy'
        elif agg_strategy == 'entropy-s+r':
            self.score_fn: Callable[[List[float]], float] = self._entropy_supp_refute
            self.suffix: str = 'entropy-sr'
        elif agg_strategy == 'neg-entropy':
            self.score_fn: Callable[[List[float]], float] = lambda probs: -1 * self._entropy(probs)
            self.suffix: str = 'neg-entropy'
        elif agg_strategy == 'neg-entropy-s+r':
            self.score_fn: Callable[[List[float]], float] = lambda probs: -1 * self._entropy_supp_refute(probs)
            self.suffix: str = 'neg-entropy-sr'
        else:
            raise NotImplementedError(agg_strategy)

    def get_suffix(self) -> str:
        return self.suffix

    def rerank(self, prediction: Dict):
        scores = []
        passages = []

        for passage_id in prediction['passage_predictions']:
            prediction_scores: List[float] = prediction['passage_predictions'][passage_id]['prediction']['probabilities']
            scores.append(self.score_fn(prediction_scores))
            passages.append(passage_id)

        scores = np.array(scores) * -1
        sorted_indices = np.argsort(scores)

        return {
            'id': prediction['id'],
            'ranked_passages': [passages[j] for j in sorted_indices],
            'scores': [scores[j] for j in sorted_indices]
        }

    def _supp_nei(self, probabilities: List[float]):
        return sum([
            probabilities[self.label2id['SUPPORT']],
            probabilities[self.label2id['NOT_ENOUGH_INFO']]
        ])

    def _entropy(self, probabilities: List[float]):
        return entropy(probabilities)

    def _entropy_supp_refute(self, probabilities: List[float]):
        return entropy(
            [probabilities[self.label2id['SUPPORT']], probabilities[self.label2id['CONTRADICT']]]
        )

    def _supp_refute(self, probabilities: List[float]):
        return sum([
            probabilities[self.label2id['SUPPORT']],
            probabilities[self.label2id['CONTRADICT']]
        ])

    def _supp(self, probabilities: List[float]):
        return probabilities[self.label2id['SUPPORT']]

    def _refute(self, probabilities: List[float]):
        return probabilities[self.label2id['CONTRADICT']]