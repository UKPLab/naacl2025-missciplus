from typing import List, Dict

import numpy as np
from sklearn.metrics import average_precision_score


def precision_at_k(gold_passages: List[str], ranked_passages: List[str], k: int) -> float:

    if len(ranked_passages) == 0:
        return 0.

    if len(gold_passages) == 0:
        raise NotImplementedError('no gold passages supplied to compute p@k')
    top_k = [
        1. if passage in gold_passages else 0.0 for passage in ranked_passages[:k]
    ]
    return sum(top_k) / len(top_k)


def reciprocal_rank(gold_passages: List[str], ranked_passages: List[str]) -> float:
    relevance = np.array([1. if passage in gold_passages else 0. for passage in ranked_passages])
    positive_positions: np.ndarray = relevance.nonzero()[0]
    if len(positive_positions) == 0:
        return 0.
    else:
        return 1. / (positive_positions[0] + 1)


def has_positive_at(gold_passages: List[str], ranked_passages: List[str], k: int) -> float:
    if len(set(gold_passages) & set(ranked_passages[:k])) > 0:
        return 1.
    else:
        return 0.


def average_precision(ranked_results: List[str], gold_label_dict: Dict[str, bool]) -> float:

    if len(ranked_results) != len(gold_label_dict):
        raise ValueError(f'Size mismatch: {len(ranked_results)} != {len(gold_label_dict)}!')

    binary_labels: List[bool] = [gold_label_dict[result] for result in ranked_results]

    # scores are just based on the position, because in our case results are already
    ranked_scores: np.ndarray = np.linspace(1, 0, len(ranked_results))
    return average_precision_score(binary_labels, ranked_scores)
