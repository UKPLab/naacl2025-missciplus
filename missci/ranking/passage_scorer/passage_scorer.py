from typing import List, Optional, Dict

import numpy as np


class BasePassageScorer:

    MEAN_POOL_SENT: str = 'mean'
    CONCAT_PASSAGE: str = 'concat'

    def __init__(self, variant: str, add_section_title: bool, k: Optional[int] = None):
        if variant not in {
            BasePassageScorer.MEAN_POOL_SENT, BasePassageScorer.CONCAT_PASSAGE
        }:
            raise ValueError(variant)

        if variant == BasePassageScorer.MEAN_POOL_SENT:
            if k is None or k < 1:
                raise ValueError(f'Must provide k with a value of 1 or higher. Found {k}')

        self.k: Optional[int] = k
        self.variant: str = variant
        self.add_section_title = add_section_title

    def _compute_score_among(self, text1: str, text2: List[str]) -> List[float]:
        raise NotImplementedError()

    def score_passage(self, reference_text: str, passage_sentences: List[str], passage_title: str) -> float:
        if self.variant == BasePassageScorer.MEAN_POOL_SENT:
            passage_sentences = list(map(
                lambda sent: self.get_passage_title_if_exists(passage_title) + sent, passage_sentences
            ))
            scores: List[float] = self._compute_score_among(reference_text, passage_sentences)
            assert len(scores) == len(passage_sentences)
            scores = sorted(scores, reverse=True)
            return float(np.mean(scores[:self.k]))
        else:
            passage_text: str = self.get_passage_title_if_exists(passage_title) + ' '.join(passage_sentences)
            scores: List[float] = self._compute_score_among(reference_text, [passage_text])
            assert len(scores) == 1
            return scores[0]

    def get_passage_title_if_exists(self, passage_title: str) -> str:
        if self.add_section_title and passage_title is not None and len(passage_title) > 0:
            return f'(Section: {passage_title}) '
        return ''

