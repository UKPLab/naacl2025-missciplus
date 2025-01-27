"""create-premise-judge2.py

Usage:
  create-premise-judge2.py baseline <model> [--claim] [--context] [--complete]
  create-premise-judge2.py zeroshot <llm> <size> <prompt> [--claim] [--context] [--complete]
  create-premise-judge2.py nli-s
  create-premise-judge2.py icl <llm> <size> <prompt> <num-samples> [--claim] [--context] [--complete]
  create-premise-judge2.py sft <llm> <size> <prompt> [--claim] [--context] [--complete] [--epochs=<epochs>] [--bs-acc=<bs-acc>] [--scheduler=<scheduler>] [--lr=<lr>] [--lora-alpha=<lora-alpha>] [--lora-dropout=<lora-dropout>] [--lora-r=<lora-r>]
  create-premise-judge2.py full-sft <llm> <size> <prompt> [--claim] [--context] [--complete] [--epochs=<epochs>] [--bs-acc=<bs-acc>] [--scheduler=<scheduler>] [--lr=<lr>] [--lora-alpha=<lora-alpha>] [--lora-dropout=<lora-dropout>] [--lora-r=<lora-r>]
"""
import json
import os
import random
from os.path import join, exists
from typing import Iterable, Dict, List, Set, Tuple, Optional

from datasets import Dataset
from docopt import docopt
from transformers import set_seed

from missci.premise_judge.instruction_bank import get_judge_instruction
from missci.premise_judge.judge_evaluator import JudgeEvaluator
from missci.premise_judge.judges.baseline_judge import BaselinePremiseJudge
from missci.premise_judge.judges.llama3_judge import Llama3Judge
from missci.premise_judge.judges.nli_s_baseline_judge import NLIBaselinePremiseJudge
from missci.premise_judge.judges.sft_llama3_judge import SFTLlama3Judge
from missci.premise_judge.premise_judge import PremiseJudge
from missci.premise_judge.xval import get_data_junks, get_xval_train_test
from missci.util.fileutil import write_json, write_jsonl


def get_addon_strings(args) -> List[str]:
    addon_str: Optional[List] = []
    if args['--claim'] and args['--context']:
        addon_str = ['connect-1']
    elif args['--claim']:
        addon_str = ['only-claim-1']
    elif args['--context']:
        addon_str = ['only-context-1']
    return addon_str


def get_judge_and_name_baseline(args) -> Tuple[PremiseJudge, str]:
    model: str = args['<model>']
    name = f'baseline-{model}'
    return BaselinePremiseJudge(model), name


def get_judge_and_name_nlis() -> Tuple[PremiseJudge, str]:
    name = f'baseline-nli-s'
    return NLIBaselinePremiseJudge(), name


def get_judge_and_name_llm(args, experiment_type: str) -> Tuple:
    # <llm> <size> <prompt>
    name: str = f'{experiment_type}-{args["<llm>"]}-{args["<size>"]}-{args["<prompt>"]}'
    if experiment_type == 'icl':
        name += f"_shots-{int(args['<num-samples>'])}"

    if args['--claim']:
        name += '-Clm'
    if args['--context']:
        name += '-Ctx'

    addon_str: Optional[List] = get_addon_strings(args)
    if args["<llm>"] == 'llama3':

        return Llama3Judge(
            setting=experiment_type,
            llama_type='llama3',
            llama_size=args["<size>"],
            instructions=get_judge_instruction(args["<prompt>"], addon_str),
            run_8bit=args["<size>"] == '70b',
            samples_per_cls=int(args['<num-samples>']) if experiment_type == 'icl' else 0,
            add_claim=args['--claim'],
            add_context=args['--context']
        ), name
    else:
        raise NotImplementedError()


def get_judge_and_name_sft_llm(args: Dict, name_prefix: str = '') -> Tuple[PremiseJudge, str]:
    llm: str = args['<llm>']
    size: str = args['<size>']
    prompt: str = args['<prompt>']
    num_epochs: int = int(args['--epochs']) if args['--epochs'] else 1
    batch_size_accum: int = int(args['--bs-acc']) if args['--bs-acc'] else 1
    scheduler: str = args['--scheduler'] if args['--scheduler'] else 'constant'
    lr: float = float(args['--lr']) if args['--lr'] else 5e-5
    lora_alpha: int = int(args['--lora-alpha']) if args['--lora-alpha'] else 16
    lora_dropout: float = float(args['--lora-dropout']) if args['--lora-dropout'] else 0.05
    lora_r: int = int(args['--lora-r']) if args['--lora-r'] else 16

    name: str = f'sft-{args["<llm>"]}-{args["<size>"]}-{args["<prompt>"]}'
    name += f'_ep{num_epochs}_ba{batch_size_accum}_sc-{scheduler}_lr{str(lr).replace(".", "")}'
    name += f'_a{lora_alpha}_drp{str(lora_dropout).replace(".", "")}_r{lora_r}'

    if args['--claim']:
        name += '-Clm'
    if args['--context']:
        name += '-Ctx'

    if llm == 'llama3':
        model: PremiseJudge = SFTLlama3Judge(
            llm=llm,
            size=size,
            instructions=get_judge_instruction(prompt_name=prompt, add_instructs=get_addon_strings(args)),
            num_epochs=num_epochs,
            batch_size_accum=batch_size_accum,
            batch_size_per_gpu=4,
            scheduler=scheduler,
            lr=lr,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_r=lora_r,
            output_name=name_prefix + name,
            add_context=args['--context'],
            add_claim=args['--claim'],
            run_8bit=size == '8b',
            run_4bit=size == '70b'
        )
        return model, name

    raise NotImplementedError()


def do_sft_on_all_data(args: Dict, file: str, directory: str = './predictions/premise-judge'):
    judge_model, base_name = get_judge_and_name_sft_llm(args, name_prefix='full-sft__')
    full_data = [
        sample for junk in get_data_junks(file) for sample in junk
    ]
    set_seed(1)
    random.shuffle(full_data)
    dataset: Dataset = Dataset.from_list(full_data)
    print('Train on:', len(dataset))
    judge_model.fit(dataset)
    print('Done.')

    print('Just for sanity!!')
    predictions: List[Dict] = judge_model.predict_dataset(dataset)
    evaluator: JudgeEvaluator = JudgeEvaluator()
    predicted_labels: List[bool] = list(map(lambda p: p['predicted'], predictions))
    evaluator.add_fold_predictions(predicted_labels, list(dataset['label']))
    evaluator.evaluate()


def do_eval(args: Dict, file: str, directory: str = './predictions/premise-judge'):
    iterations = 1
    if args['baseline']:
        judge_model, base_name = get_judge_and_name_baseline(args)
    elif args['nli-s']:
        judge_model, base_name = get_judge_and_name_nlis()
    elif args['zeroshot']:
        judge_model, base_name = get_judge_and_name_llm(args, 'zeroshot')
    elif args['icl']:
        judge_model, base_name = get_judge_and_name_llm(args, 'icl')
        iterations = 3
    elif args['sft']:
        judge_model, base_name = get_judge_and_name_sft_llm(args)
        iterations = 3
    else:
        raise NotImplementedError()

    data_junks = get_data_junks(file)
    for it in range(iterations):
        set_seed(it+1)
        evaluator: JudgeEvaluator = JudgeEvaluator()

        all_predictions: List[Dict] = []
        for dataset_dict in get_xval_train_test(data_junks):
            judge_model.fit(dataset_dict['train'])
            predictions: List[Dict] = judge_model.predict_dataset(dataset_dict['test'])
            all_predictions.extend(predictions)
            predicted_labels: List[bool] = list(map(lambda p: p['predicted'], predictions))
            evaluator.add_fold_predictions(predicted_labels, list(dataset_dict['test']['label']))
            judge_model.reset_model()

        metrics: Dict = evaluator.evaluate()
        dest_file_dir: str = join(directory, file.replace('.csv', ''))
        if not exists(dest_file_dir):
            os.makedirs(dest_file_dir)
        write_json({
            'metrics': metrics
        }, join(dest_file_dir, f'eval__{base_name}-it{it+1}.json'), pretty=True)
        write_jsonl(join(dest_file_dir, f'{base_name}-it{it + 1}.jsonl'), all_predictions)


def main():
    args = docopt(__doc__)
    for key in args:
        if isinstance(args[key], str):
            args[key] = args[key].strip()

    file: str = 'human-eval-distinct-premise.csv'
    if args['--complete']:
        file = 'human-error-analysis-COMPLETE.csv'

    if args['full-sft']:
        do_sft_on_all_data(args, file=file)
    else:
        do_eval(args, file=file)


if __name__ == '__main__':
    main()
