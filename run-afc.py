"""run-afc

Usage:
  run-afc.py train <task> <model> <out> <seed> <lr> <batch_size>
  run-afc.py inference <task> <model> [--add-p0=<add-p0>] [--add-p0-as-passage] [--seed=<seed>]
  run-afc.py evaluate <task> <file>
  run-afc.py eval-missciplus <models_dir>
"""
from typing import Dict, Any, Optional

from docopt import docopt

from missci.modeling.afc_inference import afc_inference
from missci.modeling.afc_training import train_and_eval_afc

MODEL_OUTPUT_DIR_AFC: str = './models/afc'
PREDICTION_DIRECTORY_AFC: str = './predictions/afc'
PREDICTION_DIRECTORY_MISSCI: str = './predictions/afc-missci'


def run_train_and_eval_afc(args: Dict[str, str]):
    train_and_eval_afc(
        task=args['<task>'],
        model_name=args['<model>'],
        out_name=args['<out>'],
        seed=int(args['<seed>']),
        lr=float(args['<lr>']),
        batch_size=int(args['<batch_size>']),
        dest_dir=MODEL_OUTPUT_DIR_AFC
    )


def run_afc_inference(args: Dict[str, Any]):
    target_task: str = args['<task>']
    model_name: str = args['<model>']
    add_p0: str = args['--add-p0']
    add_p0_as_passage: bool = args['--add-p0-as-passage']
    seed: Optional[int] = int(args['--seed']) if args['--seed'] is not None else None

    afc_inference(
        model_name=model_name,
        target_task=target_task,
        prediction_directory=PREDICTION_DIRECTORY_AFC if target_task != 'missci' else PREDICTION_DIRECTORY_MISSCI,
        model_directory=MODEL_OUTPUT_DIR_AFC,
        add_p0_to=add_p0,
        add_p0_as_passage=add_p0_as_passage,
        seed=seed
    )


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['train']:
        run_train_and_eval_afc(args)
    elif args['inference']:
        run_afc_inference(args)
    else:
        raise NotImplementedError()
