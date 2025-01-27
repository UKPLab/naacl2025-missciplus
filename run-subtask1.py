"""run-subtask1-baselines

Usage:
  run-subtask1.py bm25 [--fullstudy] [--dev]
  run-subtask1.py random [--fullstudy] [--dev]
  run-subtask1.py ordered [--fullstudy] [--dev]
  run-subtask1.py sentence-embeddings <model> <variant> [--k=<k>] [--fullstudy] [--dev] [--add-title]
  run-subtask1.py afc-eval <prediction-directory-name> <file-name> [--fullstudy] [--dev] [--add-title]


Options:
  --destination_directory <destination>  Destination directory for model output (optional)
  --data_directory <data>               Data directory (optional)
  -h, --help                           Show this help message and exit
"""
from os import listdir
from os.path import join
from typing import List, Dict, Optional

import numpy as np
from docopt import docopt

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.evaluator_subtask1 import Subtask1Evaluator
from missci.ranking.afc_reranker import AFCReranker
from missci.ranking.bm25 import run_bm25_baseline
from missci.ranking.embedding_ranker import EmbeddingRanker
from missci.ranking.naive_baselines import run_ordered_baseline, run_random_baseline
from missci.util.directory_util import get_prediction_directory
from missci.util.fileutil import read_jsonl, write_jsonl

PREDICTION_DIRECTORY_AFC: str = './predictions/afc-missci'


def run_bm25(args: Dict, split: str, instances: List[Dict], use_full_study: bool) -> str:

    return run_bm25_baseline(
        get_prediction_directory('subtask1'), instances,
        use_full_study=use_full_study, init_with_full_study=use_full_study, prefix=f'{split}__'
    )


def baseline_ordered(split: str, instances: List[Dict], use_full_study: bool) -> str:
    return run_ordered_baseline(get_prediction_directory('subtask1'), instances, use_full_study, prefix=f'{split}_')


def baseline_random(split: str, instances: List[Dict], use_full_study: bool) -> List[str]:
    return [
        run_random_baseline(get_prediction_directory('subtask1'), instances, seed, use_full_study, prefix=f'{split}_')
        for seed in range(1, 6)
    ]


def run_sentence_embedding_ranker(
        args: Dict, split: str, instances: List[Dict], use_full_study: bool
) -> str:
    embedder: str = args['<model>']
    embd_variant: str = args['<variant>']
    add_title: bool = args['--add-title']
    k: Optional[int] = None
    if args['--k'] is not None:
        k = int(args['--k'])

    ranker: EmbeddingRanker = EmbeddingRanker(
        embedder, embd_variant, get_prediction_directory('subtask1'), k, add_title
    )
    return ranker.run(instances, use_full_study, f'{split}_')


def make_afc_ranking(prediction_directory_name: str, file_name: str) -> List[str]:
    current_afc_directory: str = join(PREDICTION_DIRECTORY_AFC, prediction_directory_name)
    model_names: List[str] = listdir(current_afc_directory)

    if len(model_names) != 5:
        raise ValueError(f'Expected 5 model files but found {len(model_names)}!')

    reranker: AFCReranker = AFCReranker(agg_strategy='s', task_name=prediction_directory_name)
    prediction_names: List[str] = []
    for model_name in model_names:
        predictions: List[Dict] = list(read_jsonl(join(join(current_afc_directory, model_name), file_name)))
        predictions = list(map(reranker.rerank, predictions))
        out_name: str = f'afc-rerank_{model_name}__{file_name}'.replace('.jsonl', f'.supported.jsonl')
        write_jsonl(join(get_prediction_directory('subtask1'), out_name), predictions)
        prediction_names.append(out_name)

    return prediction_names


def main():
    args = docopt(__doc__)

    split = 'dev' if args['--dev'] else 'test'
    use_full_study: bool = '--fullstudy' in args and args['--fullstudy']
    instances: List[Dict] = MappedDataLoader().load_raw_arguments(split)

    prediction_files: List[str] = []

    if args['bm25']:
        prediction_files.append(run_bm25(args, split, instances, use_full_study))
    elif args['random']:
        prediction_files.extend(baseline_random(split, instances, use_full_study))
    elif args['ordered']:
        prediction_files.append(baseline_ordered(split, instances, use_full_study))
    elif args['sentence-embeddings']:
        prediction_files.append(run_sentence_embedding_ranker(args, split, instances, use_full_study))
    elif args['afc-eval']:
        # This assumes to have inference predictions on the argument level already!
        prediction_files.extend(make_afc_ranking(args['<prediction-directory-name>'], args['<file-name>']))
    else:
        raise NotImplementedError()

    if len(prediction_files) == 0:
        print('No prediction files exist yet!')
    else:
        evaluator: Subtask1Evaluator = Subtask1Evaluator(split=split, use_full_study=use_full_study)
        scores = [
            evaluator.evaluate_file(prediction_file) for prediction_file in prediction_files
        ]

        keys = list(scores[0].keys())
        for key in keys:
            if key != 'experiment_data':
                print(key, round(float(np.mean([score[key] for score in scores])), 3))


if __name__ == '__main__':
    main()
