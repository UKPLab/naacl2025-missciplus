"""subtask2_open

Usage:
  run-subtask2-open.py random [--dev]
  run-subtask2-open.py ordered [--dev]
  run-subtask2-open.py bm25 <seed> [--p0] [--dev]
  run-subtask2-open.py sentence-embeddings <model> <variant> <seed> [--k=<k>] [--p0] [--dev]
  run-subtask2-open.py afc-eval <task-name> <strategy> [--add-p0=<add-p0>] [--dev]
  run-subtask2-open.py llama <prompt-template> <model-size> [--dev]
  run-subtask2-open.py parse-prompt <file> [--dev]


Options:
  --destination_directory <destination>  Destination directory for model output (optional)
  --data_directory <data>               Data directory (optional)
  -h, --help                           Show this help message and exit
"""
import random
from copy import deepcopy
from os import listdir
from os.path import join
from typing import List, Dict, Optional

import numpy as np
from docopt import docopt

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.evaluator_subtask2_open import Subtask2EvaluatorOpen
from missci.ranking.afc_reranker import AFCReranker
from missci.ranking.bm25 import run_bm25_baseline
from missci.ranking.embedding_ranker import EmbeddingRanker
from missci.ranking.naive_baselines import run_random_baseline, run_ordered_baseline
from missci.util.directory_util import get_prediction_directory
from missci.util.fileutil import read_jsonl, write_jsonl
from missci.util.passage_util import get_p0_from_instance


def add_p0_passage_to_claim(instance: Dict) -> Dict:
    instance = deepcopy(instance)
    claim: str = instance['argument']['claim']
    instance['argument']['original_claim'] = claim

    p0: str = get_p0_from_instance(instance, add_p0_as_passage=True)
    instance['argument']['claim'] = p0 + ' Therefore: ' + claim
    return instance


def run_select_random(instances: List[Dict], split: str) -> List[str]:
    return [
        run_random_baseline(get_prediction_directory(
            'subtask2-open'), instances, seed, use_full_study=True, prefix=f'{split}_'
        )
        for seed in range(1, 6)
    ]


def run_select_ordered(instances: List[Dict], split: str) -> str:
    return run_ordered_baseline(
        get_prediction_directory('subtask2-open'), instances, use_full_study=True, prefix=f'{split}_'
    )


def run_bm25(instances: List[Dict], split: str, add_p0: bool, random_seed: int) -> str:
    random.seed(random_seed)

    if add_p0:
        prefix: str = f'{split}__add-P0__'
        instances = list(map(add_p0_passage_to_claim, instances))
    else:
        prefix: str = f'{split}__'

    prefix += f's{random_seed}__'

    return run_bm25_baseline(
        get_prediction_directory('subtask2-open'), instances,
        use_full_study=True, init_with_full_study=True, prefix=prefix
    )


def run_sentence_embedding_ranker(
        args: Dict, split: str, instances: List[Dict], add_p0: bool, random_seed: int
) -> str:
    random.seed(random_seed)
    if add_p0:
        prefix: str = f'{split}__add-P0__'
        instances = list(map(add_p0_passage_to_claim, instances))
    else:
        prefix: str = f'{split}__'

    prefix += f's{random_seed}__'

    embedder: str = args['<model>']
    embd_variant: str = args['<variant>']
    k: Optional[int] = None
    if args['--k'] is not None:
        k = int(args['--k'])
    ranker: EmbeddingRanker = EmbeddingRanker(embedder, embd_variant, get_prediction_directory('subtask2-open'), k)
    return ranker.run(instances, use_full_study=True, prefix=prefix)


def make_afc_ranking(task_name: str, add_p0: Optional[str], agg_strategy: str, split: str) -> List[str]:

    if add_p0 == 'claim-passage':
        p0_variant: str = '.p0-claim-passage'
    elif add_p0 == 'evidence-passage':
        p0_variant: str = '.p0-evidence-passage'
    else:
        if add_p0 is not None:
            raise ValueError(f'Unknown variant: {add_p0}!')
        p0_variant: str = ''

    file_name: str = f'{split}__missci{p0_variant}.args.jsonl'

    afc_directory: str = './predictions/afc'
    current_afc_directory: str = join(afc_directory, task_name)
    model_names: List[str] = listdir(current_afc_directory)
    if len(model_names) != 5:
        raise ValueError(f'Expected 5 model files but found {len(model_names)}!')

    re_ranker: AFCReranker = AFCReranker(agg_strategy=agg_strategy, task_name=task_name)
    prediction_names: List[str] = []
    for model_name in model_names:
        predictions: List[Dict] = list(read_jsonl(join(join(current_afc_directory, model_name), file_name)))
        predictions = list(map(re_ranker.rerank, predictions))
        out_name: str = f'afc-rerank_{model_name}__{file_name}'.replace('.jsonl', f'.{re_ranker.get_suffix()}.jsonl')
        write_jsonl(join(get_prediction_directory('subtask2-open'), out_name), predictions)
        prediction_names.append(out_name)

    return prediction_names


def main():
    args = docopt(__doc__)

    split = 'dev' if args['--dev'] else 'test'
    instances: List[Dict] = MappedDataLoader().load_raw_arguments(split)
    prediction_files: List[str] = []

    if args['random']:
        prediction_files.extend(run_select_random(instances, split))
    elif args['ordered']:
        prediction_files.append(run_select_ordered(instances, split))
    elif args['bm25']:
        prediction_files.append(run_bm25(instances, split, add_p0=args['--p0'], random_seed=int(args['<seed>'])))
    elif args['sentence-embeddings']:
        prediction_files.append(run_sentence_embedding_ranker(
            args, split, instances, add_p0=args['--p0'], random_seed=int(args['<seed>']))
        )
    elif args['afc-eval']:
        prediction_files.extend(make_afc_ranking(args['<task-name>'], args['--add-p0'], args['<strategy>'], split))
    else:
        raise NotImplementedError()

    if len(prediction_files) == 0:
        print('No prediction files exist yet!')
    else:
        evaluator: Subtask2EvaluatorOpen = Subtask2EvaluatorOpen(split=split)
        scores = [
            evaluator.evaluate_file(prediction_file) for prediction_file in prediction_files
        ]

        keys = list(scores[0].keys())
        print('Predictions on', split, ':')
        for key in keys:
            if key != 'experiment_data':
                print(key, round(float(np.mean([score[key] for score in scores])), 3))

        line = ''
        for key in [
            'mAP', 'P@1', 'P@3', 'P@10', 'R@1', 'R@3', 'R@10'
        ]:
            line += f' & {round(float(np.mean([score[key] for score in scores])), 3)}'
        print(line + '\\\\')


if __name__ == '__main__':
    main()

