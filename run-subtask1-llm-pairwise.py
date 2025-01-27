"""run-subtask1-llm-pairwise

Usage:
  run-subtask1-llm-pairwise.py llama2 <prompt-template> <model-size> <num_it> <initial-order> <seed> [--temperature=<temperature>] [--dev] [--8bit] [--add-section]
  run-subtask1-llm-pairwise.py llama3 <prompt-template> <model-size> <num_it> <initial-order> <seed> [--temperature=<temperature>] [--dev] [--8bit] [--add-section]
  run-subtask1-llm-pairwise.py chatgpt <prompt-template> <num_it> <initial-order> <num> [--dev] [--add-section]

Options:
  --destination_directory <destination>  Destination directory for model output (optional)
  --data_directory <data>               Data directory (optional)
  -h, --help                           Show this help message and exit
"""
import json
import random
from os.path import join
from typing import List, Dict, Optional

from docopt import docopt
from tqdm import tqdm

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.evaluator_subtask1 import Subtask1Evaluator
from missci.modeling.basic_llm.basic_chatgpt import BasicAnyGPT
from missci.modeling.basic_llm.basic_llama2 import BasicLlama2
from missci.modeling.basic_llm.basic_llama3 import BasicLlama3Pipeline
from missci.prompt_generators.pairwise.basic_prp_filler import BasicPRPFiller

from missci.ranking.prp_sliding import PRPSliding
from missci.util.directory_util import get_prediction_directory
from missci.util.fileutil import write_jsonl


def create_prediction_after_k_iterations(prediction: Dict, num_it: int) -> Dict:
    return {
        'id': prediction['id'],
        'ranked_passages': prediction['ranked_passages'][num_it],
        'experiment_data': prediction['experiment_data'] | {'num_iterations': num_it + 1}
    }


def run_llama3_ranking(
        model_size: str, prompt_template: str, split: str, instances: List[Dict], num_iterations: int,
        initial_ordering: str, run_8bit: bool, add_section_title: bool,
        seed: int, temperature: Optional[float]
):

    info_str: str = ''
    if run_8bit:
        info_str += '-8bit'
    if add_section_title:
        info_str += '-sec'

    dest_file_suffix: str = f'{model_size}{info_str}-{initial_ordering}-s{seed}-t{temperature}.{split}'

    prp_sliding: PRPSliding = PRPSliding(
        num_iterations=num_iterations,
        initial_order=initial_ordering,
        overwrite=True,
        suffix=dest_file_suffix,
        llm=BasicLlama3Pipeline(
            llama_size=model_size, run_8bit=run_8bit, temperature=temperature
        ),
        template_filler=BasicPRPFiller(
            prompt_template, 'llama3', add_section_title=add_section_title, convert_prompt_format=False
        ),
        random_seed=seed
    )

    predictions: List[Dict] = []
    for instance in instances:
        predictions.append(prp_sliding.rank_passages(instance=instance))

    for i in range(num_iterations):
        iteration_predictions: List[Dict] = list(
            map(lambda pred: create_prediction_after_k_iterations(pred, i), predictions)
        )

        current_prediction_file_name: str = prp_sliding.get_name(i+1)
        write_jsonl(join(get_prediction_directory('subtask1'), current_prediction_file_name), iteration_predictions)

        evaluator: Subtask1Evaluator = Subtask1Evaluator(split=split, use_full_study=False)
        scores = evaluator.evaluate_file(current_prediction_file_name)
        print(f'Scores after it={i+1}:')
        print(json.dumps(scores, indent=2))


def run_llama_ranking(
        model_size: str, prompt_template: str, split: str, instances: List[Dict], num_iterations: int,
        initial_ordering: str, run_8bit: bool, add_section_title: bool,
        seed: int, temperature: Optional[float]
):

    info_str: str = ''
    if run_8bit:
        info_str += '-8bit'
    if add_section_title:
        info_str += '-sec'

    dest_file_suffix: str = f'{model_size}{info_str}-{initial_ordering}-s{seed}-t{temperature}.{split}'

    prp_sliding: PRPSliding = PRPSliding(
        num_iterations=num_iterations,
        initial_order=initial_ordering,
        overwrite=True,
        suffix=dest_file_suffix,
        llm=BasicLlama2(
            llama_size=model_size, run_8bit=run_8bit, temperature=temperature
        ),
        template_filler=BasicPRPFiller(prompt_template, 'llama2', add_section_title=add_section_title),
        random_seed=seed
    )

    predictions: List[Dict] = []
    for instance in instances:
        predictions.append(prp_sliding.rank_passages(instance=instance))

    for i in range(num_iterations):
        iteration_predictions: List[Dict] = list(
            map(lambda pred: create_prediction_after_k_iterations(pred, i), predictions)
        )

        current_prediction_file_name: str = prp_sliding.get_name(i+1)
        write_jsonl(join(get_prediction_directory('subtask1'), current_prediction_file_name), iteration_predictions)

        evaluator: Subtask1Evaluator = Subtask1Evaluator(split=split, use_full_study=False)
        scores = evaluator.evaluate_file(current_prediction_file_name)
        print(f'Scores after it={i+1}:')
        print(json.dumps(scores, indent=2))


def run_gpt_ranking(
        prompt_template: str, split: str, instances: List[Dict], num_iterations: int,
        initial_ordering: str, add_section_title: bool, num: int
):
    info_str: str = ''
    if add_section_title:
        info_str += '-sec'

    dest_file_suffix: str = f'{info_str}-{initial_ordering}.{split}-num{num}'

    prp_sliding: PRPSliding = PRPSliding(
        num_iterations=num_iterations,
        initial_order=initial_ordering,
        overwrite=False,
        suffix=dest_file_suffix,
        llm=BasicAnyGPT(),
        template_filler=BasicPRPFiller(prompt_template, 'chatgpt', add_section_title=add_section_title)
    )
    random.seed(1)
    predictions: List[Dict] = []
    for instance in tqdm(instances):
        predictions.append(prp_sliding.rank_passages(instance=instance))

    for i in range(num_iterations):
        iteration_predictions: List[Dict] = list(
            map(lambda pred: create_prediction_after_k_iterations(pred, i), predictions)
        )

        current_prediction_file_name: str = prp_sliding.get_name(i + 1)
        write_jsonl(join(get_prediction_directory('subtask1'), current_prediction_file_name), iteration_predictions)

        evaluator: Subtask1Evaluator = Subtask1Evaluator(split=split, use_full_study=False)
        scores = evaluator.evaluate_file(current_prediction_file_name)
        print(f'Scores after it={i + 1}:')
        print(json.dumps(scores, indent=2))


def main():
    args = docopt(__doc__)

    split = 'dev' if args['--dev'] else 'test'
    instances: List[Dict] = MappedDataLoader().load_raw_arguments(split)

    if args['llama2']:
        run_llama_ranking(
            args['<model-size>'], args['<prompt-template>'], split, instances, int(args['<num_it>']),
            args['<initial-order>'], args['--8bit'], args['--add-section'],
            int(args['<seed>']),
            float(args['--temperature']) if args['--temperature'] is not None else None
        )
    elif args['llama3']:
        run_llama3_ranking(
            args['<model-size>'], args['<prompt-template>'], split, instances, int(args['<num_it>']),
            args['<initial-order>'], args['--8bit'], args['--add-section'],
            int(args['<seed>']),
            float(args['--temperature']) if args['--temperature'] is not None else None
        )
    elif args['chatgpt']:
        run_gpt_ranking(
            args['<prompt-template>'], split, instances, int(args['<num_it>']),
            args['<initial-order>'], args['--add-section'], args['<num>']
        )
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()

