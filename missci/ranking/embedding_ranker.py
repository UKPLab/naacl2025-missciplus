from os.path import join
from typing import Dict, Optional, List, Tuple

from tqdm import tqdm

from missci.ranking.passage_scorer.cosine_passage_scorer import CosineTextPassageScorer
from missci.ranking.passage_scorer.ims_passage_scorer import IMSPassageScorer
from missci.ranking.passage_scorer.instructor_passage_scorer import InstructPassageScorer
from missci.ranking.passage_scorer.passage_scorer import BasePassageScorer
from missci.util.fileutil import write_jsonl


def get_key_to_model() -> Dict[str, str]:
    return {
        'sbert': 'all-mpnet-base-v2',
        'biobert-st': 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
        'sapbert-st': 'pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb',
        'pubmedbert-st': 'pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
    }


def get_text_passage_scorer(
        sentence_embedder_key: str, variant: str, k: Optional[int], add_section_title: bool
) -> BasePassageScorer:
    if sentence_embedder_key in {'sbert', 'biobert-st', 'sapbert-st', 'pubmedbert-st'}:
        return CosineTextPassageScorer(
            variant, get_key_to_model()[sentence_embedder_key], k=k, add_section_title=add_section_title
        )
    elif sentence_embedder_key in {'instructor', 'instructor-undermine', 'instructor-refute'}:

        query_prompts_claim: Dict[str, str] = {
            'scientific_sent': 'Represent the scientific claim for retrieving supporting sentences: ',
            'scientific_passage': 'Represent the scientific claim for retrieving supporting passages: '
        }

        query_prompts_document: Dict[str, str] = {
            'scientific_sent': 'Represent the scientific sentence for retrieval: ',
            'scientific_passage': 'Represent the scientific passage for retrieval: '
        }

        if variant == 'mean':
            prompt_key: str = 'scientific_sent'
        elif variant == 'concat':
            prompt_key: str = 'scientific_passage'
        else:
            raise NotImplementedError(variant)

        query_prompt: str = query_prompts_claim[prompt_key]
        if sentence_embedder_key == 'instructor-undermine':
            query_prompt = query_prompt.replace('supporting', 'undermining')
        elif sentence_embedder_key == 'instructor-refute':
            query_prompt = query_prompt.replace('supporting', 'refuting')

        return InstructPassageScorer(
            variant, query_prompt=query_prompt, document_prompt=query_prompts_document[prompt_key], k=k,
            add_section_title=add_section_title
        )

    elif sentence_embedder_key in {'ims'}:
        return IMSPassageScorer(variant, None, k=k, add_section_title=add_section_title)
    else:
        raise NotImplementedError(sentence_embedder_key)


class EmbeddingRanker:
    def __init__(
            self, sentence_embedder_key: str, variant: str,
            dest_directory: str, k: Optional[int] = 1, add_title: bool = False
    ):

        self.variant: str = variant
        self.k = k
        self.sentence_embedder_key: str = sentence_embedder_key
        self.scorer: BasePassageScorer = get_text_passage_scorer(sentence_embedder_key, variant, k, add_title)
        self.add_title: bool = add_title
        self.name: str = f'embd_{sentence_embedder_key}_{variant}{"-sec" if add_title else ""}.jsonl'
        if k is not None:
            self.name = self.name.replace('.jsonl', f'_k{k}.jsonl')
        self.dest_directory: str = dest_directory

    def run(self, instances: List[Dict], use_full_study: bool, prefix: str) -> str:
        if use_full_study:
            dest_name: str = self.name.replace('.jsonl', '.fullstudy.jsonl')
            use_passage_key: str = 'all_passages'
        else:
            dest_name = self.name[:]
            use_passage_key: str = 'selected_passages'

        dest_name = prefix + dest_name

        predictions: List[Dict] = []
        for instance in tqdm(instances):
            claim: str = instance['argument']['claim']

            passages: Dict[str, Dict] = instance['study'][use_passage_key]
            scored_passages: [List[Tuple[str, float]]] = []
            for passage_key in passages:
                sentences: List[str] = passages[passage_key]['sentences']
                passage_title: str = passages[passage_key]['section']
                score: float = self.scorer.score_passage(claim, sentences, passage_title)
                scored_passages.append((passage_key, score))

            scored_passages = sorted(scored_passages, key=lambda x: -x[-1])
            passages, scores = map(list, zip(*scored_passages))
            predictions.append({
                'id': instance['id'],
                'ranked_passages': passages,
                'cosine_similarities': scores,
                'experiment_data': {
                    'sentence_embedder_key': self.sentence_embedder_key,
                    'variant': self.variant,
                    'k': self.k
                }
            })

        write_jsonl(join(self.dest_directory, dest_name), predictions)
        return dest_name





