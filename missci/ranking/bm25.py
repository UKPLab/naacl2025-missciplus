from os.path import join
from typing import List, Dict, Iterable

import spacy
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from missci.util.fileutil import write_jsonl
from missci.util.passage_util import get_sorted_passages


class BM25Preprocessor:
    def __init__(self, lowercase: bool = True, spacy_version: str = 'en_core_web_sm'):
        self.lowercase: bool = lowercase
        self.nlp = spacy.load(spacy_version)

    def get_tokens(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens: Iterable[str] = map(lambda x: x.text, doc)
        if self.lowercase:
            tokens = map(lambda x: x.lower(), tokens)
        return list(tokens)


class BM25Scorer:
    def __init__(self,
                 text_preprocessor: BM25Preprocessor,
                 instances: List[Dict],
                 include_full_studies: bool
                 ):
        self.text_preprocessor: BM25Preprocessor = text_preprocessor

        # Instantiate the corpus
        self.passage_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

        for instance in tqdm(instances):
            passage_keys: Iterable[str] = get_sorted_passages(instance, from_full_study=include_full_studies)

            # Because some studies are used multiple times.
            passage_keys = filter(lambda x: x not in self.passage_ids, passage_keys)

            for passage_key in passage_keys:
                self.passage_ids.append(passage_key)
                passage_text: str = ' '.join(instance['study']['all_passages'][passage_key]['sentences'])
                self.tokenized_corpus.append(self.text_preprocessor.get_tokens(passage_text))

        print('Dataset contains', len(self.passage_ids), 'distinct passages.')
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_ranked_results(self, text: str, passage_id_candidates: List[str]) -> List[Dict]:
        preprocessed_text: List[str] = self.text_preprocessor.get_tokens(text)
        scores = list(map(float, self.bm25.get_scores(preprocessed_text)))
        keep_indices: List[int] = list(
            filter(lambda i: self.passage_ids[i] in passage_id_candidates, range(len(self.passage_ids)))
        )

        assert len(passage_id_candidates) == len(keep_indices)
        results: List[Dict] = [
            {'passage': self.passage_ids[keep_idx], 'score': scores[keep_idx]}
            for keep_idx in keep_indices
        ]
        return sorted(results, key=lambda x: -x['score'])


def run_bm25_baseline(dest_dir: str, instances: List[Dict], use_full_study: bool, init_with_full_study: bool, prefix: str):

    bm25: BM25Scorer = BM25Scorer(
        BM25Preprocessor(), instances, include_full_studies=init_with_full_study
    )

    predictions: List[Dict] = []
    for instance in tqdm(instances):
        passage_ids: List[str] = get_sorted_passages(instance, from_full_study=use_full_study)
        ranked: List[Dict] = bm25.get_ranked_results(instance['argument']['claim'], passage_ids)

        ranked_passage_ids: List[str] = list(map(lambda x: x['passage'], ranked))
        ranked_passage_scores: List[float] = list(map(lambda x: x['passage'], ranked))

        predictions.append({
            'id': instance['id'],
            'ranked_passages': ranked_passage_ids,
            'scores': ranked_passage_scores
        })

    file_name: str = f'baseline_bm25.jsonl'
    if use_full_study:
        file_name = file_name.replace('.jsonl', '.fullstudy.jsonl')

    if init_with_full_study:
        file_name = file_name.replace('.jsonl', '.ifs.jsonl')  # init with full study
    else:
        file_name = file_name.replace('.jsonl', '.isp.jsonl')  # init with selected passages

    file_name = prefix + file_name

    write_jsonl(join(dest_dir, file_name), predictions)
    return file_name
