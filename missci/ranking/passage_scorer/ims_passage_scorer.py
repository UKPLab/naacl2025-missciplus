from typing import List, Optional

import numpy as np

from scientific_information_change.estimate_similarity import SimilarityEstimator

from missci.ranking.passage_scorer.passage_scorer import BasePassageScorer


class IMSPassageScorer(BasePassageScorer):

    def __init__(
            self, variant: str, model_name: Optional[str] = None, add_section_title: bool = False, k: Optional[int] = None
    ):
        super().__init__(variant, add_section_title, k=k)

        if model_name is not None:
            self.model: SimilarityEstimator = SimilarityEstimator(model_name)
        else:
            self.model: SimilarityEstimator = SimilarityEstimator()

    def _compute_score_among(self, text1: str, text2: List[str]) -> List[float]:
        result: np.ndarray = self.model.estimate_ims(a=[text1], b=text2)
        return list(map(float, result[0]))

