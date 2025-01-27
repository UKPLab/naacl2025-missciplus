from typing import Optional, List

from InstructorEmbedding import INSTRUCTOR

from sklearn.metrics.pairwise import cosine_similarity

from missci.ranking.passage_scorer.passage_scorer import BasePassageScorer


class InstructPassageScorer(BasePassageScorer):

    def __init__(
            self,
            variant: str,
            query_prompt: str,
            document_prompt: str,
            add_section_title: bool,
            k: Optional[int] = None
    ):
        super().__init__(variant, add_section_title, k=k)
        self.query_prompt: str = query_prompt
        self.document_prompt: str = document_prompt
        self.model = INSTRUCTOR('hkunlp/instructor-xl')

    def _compute_score_among(self, text1: str, text2: List[str]) -> List[float]:
        query_embeddings = self.model.encode([[self.query_prompt, text1]])
        corpus_embeddings = self.model.encode([
            [self.document_prompt, sentence] for sentence in text2
        ])
        return list(map(float, cosine_similarity(query_embeddings, corpus_embeddings).reshape(-1).tolist()))
