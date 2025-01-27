from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer, util

from missci.ranking.passage_scorer.passage_scorer import BasePassageScorer


class CosineTextPassageScorer(BasePassageScorer):

    def __init__(self, variant: str, model_name: str, add_section_title: bool, k: Optional[int] = None):
        super().__init__(variant, add_section_title, k=k)
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def _compute_score_among(self, text1: str, text2: List[str]) -> List[float]:
        sentences: List[str] = [text1] + text2
        embeddings: torch.Tensor = self.model.encode(sentences, convert_to_tensor=True)
        embedding_text1: torch.Tensor = embeddings[0]
        embeddings_text2: torch.Tensor = embeddings[1:]
        return util.cos_sim(embedding_text1, embeddings_text2)[0].tolist()
