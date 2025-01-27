"""run-subtask-3.py

Usage:
  run-subtask-3.py passage-wise <llm> <template> <seed> [--temperature=<temperature>] [--dev] [--add-section] [--small]
  run-subtask-3.py concat-passages-correct <llm> <template> <seed> [<file>] [--temperature=<temperature>] [--small]
  run-subtask-3.py concat-passages <llm> <template> <seed> <order> [--temperature=<temperature>] [--dev] [--add-section] [--small] [--all]
  run-subtask-3.py concat-context <llm> <template> <seed> [--temperature=<temperature>] [--dev] [--small]
  run-subtask-3.py context-wise <llm> <template> <seed> [--temperature=<temperature>] [--dev] [--small]
  run-subtask-3.py parse <file> [--dev] [--file=<file>]
  run-subtask-3.py map <file> [--dev] [--overwrite]
  run-subtask-3.py evaluate <file> [--dev] [--all] [--only-mapped-context]
  run-subtask-3.py evaluate-binary <file> [--dev] [--file=<file>]

"""
import random
import shutil
from collections import defaultdict
from os.path import join, exists
from typing import List, Dict, Optional, Iterable, Set, Union

from docopt import docopt
from tqdm import tqdm
from transformers import set_seed

from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.evaluator_gen_classify_missciplus import GenClassifyEvaluatorMissciPlus
from missci.modeling.basic_llm.basic_llama2 import BasicLlama2
from missci.modeling.basic_llm.basic_llama3 import BasicLlama3Pipeline
from missci.modeling.basic_llm.llm import LLM
from missci.output_parser.fallacy_mapper import FallacyMapper
from missci.output_parser.llm_output_parser_argument_reconstruction import ArgumentReconstructionParser
from missci.premise_judge.argument_fallacy_mapper import ArgumentFallacyMapper
from missci.premise_judge.instruction_bank import get_judge_instruction
from missci.premise_judge.judges.sft_llama3_judge import SFTLlama3Judge
from missci.prompt_generators.afc_for_fallacy_prompt_generator import AFCFallacyPromptGenerator
from missci.prompt_generators.prompt_generator_concat_context import ConcatContextPromptGenerator
from missci.prompt_generators.prompt_generator_concat_passages import ConcatPassagesPromptGenerator
from missci.util.directory_util import get_raw_prompt_prediction_directory, get_prediction_directory
from missci.util.fileutil import read_text, write_jsonl, read_jsonl, write_json
from missci.util.passage_util import get_p0_passages, get_sorted_passages, get_passage_text


def make_fallacy_prompt_complete(template: str, claim: str, p0: str, context: str) -> str:
    assert '@@claim@@' in template
    assert '@@p0@@' in template
    assert '@@context@@' in template
    prompt = template.replace(
        '@@claim@@', claim
    ).replace(
        '@@context@@', context
    ).replace('@@p0@@', p0).replace(
        '@@system_prompt@@', ''
    )
    assert '@@' not in prompt
    return prompt


def run_concat_context_experiments(
        llm: LLM, instances: List[Dict], template_file: str, seed: int, split: str
):
    set_seed(seed)
    template_name: str = template_file.replace('/', '--').replace('\\', '--').replace('.txt', '')
    prompt_generator: ConcatContextPromptGenerator = ConcatContextPromptGenerator(template_file)

    predictions: List[Dict] = []
    for instance in tqdm(instances):
        prompt_data: Dict = prompt_generator.make_prompt_for_instance(instance)
        output: Dict = llm.prompt_with_special_tokens(prompt_data['prompt'])
        prompt_data['params'] = {
            'split': split, 'llm': llm.get_specs(), 'template': template_file, 'seed': seed
        }
        prompt_data['output'] = output
        predictions.append(prompt_data)

    file_name: str = f'st3-concat-ctx__{template_name}__{llm.specs_string()}-s{seed}.{split}.jsonl'
    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    write_jsonl(join(directory, file_name), predictions)


def run_concat_passages_true_claims_experiments(
            llm: LLM, data_file: str, template_file: str, seed: int, file_directory: str = './afc_data/'
        ):
    set_seed(seed)
    template_name: str = template_file.replace('/', '--').replace('\\', '--').replace('.txt', '')

    instances: List[Dict] = list(read_jsonl(join(file_directory, data_file)))
    prompt_generator: AFCFallacyPromptGenerator = AFCFallacyPromptGenerator(template_file)

    predictions: List[Dict] = []
    for instance in tqdm(instances):
        prompt_data: Dict = prompt_generator.make_prompt_for_instance(instance)
        output: Dict = llm.prompt_with_special_tokens(prompt_data['prompt'])
        prompt_data['params'] = {
            'data_file': data_file, 'llm': llm.get_specs(), 'template': template_file, 'seed': seed
        }
        prompt_data['output'] = output
        predictions.append(prompt_data)

    file_name: str = f'st3-concat-true__{template_name}__{llm.specs_string()}-s{seed}.{data_file.replace(".jsonl", "")}.jsonl'
    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    write_jsonl(join(directory, file_name), predictions)


def run_concat_passages_true_experiments(
        llm: LLM, file_name: str, template_file: str, seed: int
):

    def make_prompt(use_template: str, item: Dict) -> str:
        use_prompt: str = use_template.replace(
            '@@claim@@', item['claim']
        ).replace(
            '@@context@@', item['evidence']
        ).replace(
            '@@system_prompt@@', ''
        )

        assert '@@' not in use_prompt
        return use_prompt

    file_name = file_name or 'cov-health-true-50.jsonl'
    template: str = read_text(join('./prompt_templates', template_file))
    set_seed(seed)
    predictions: List[Dict] = []
    template_name: str = template_file.replace('/', '--').replace('\\', '--').replace('.txt', '')
    for instance in tqdm(list(read_jsonl(join('./afc_data', file_name)))):
        prompt: str = make_prompt(template, instance)
        prompt_data = {
            'prompt': prompt,
            'params': {
                'llm': llm.get_specs(), 'template': template_file, 'seed': seed
            },
            'instance': instance
        }
        output: Dict = llm.prompt_with_special_tokens(prompt_data['prompt'])
        prompt_data['output'] = output
        predictions.append(prompt_data)

    file_name: str = f'st3-concat-true__{template_name}__{llm.specs_string()}-s{seed}.{file_name}.jsonl'
    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    write_jsonl(join(directory, file_name), predictions)


def run_concat_passages_experiments(
        llm: LLM, instances: List[Dict], template_file: str, add_section_title: bool,
        seed: int, split: str, passage_order: str, use_all_passages: bool
):

    set_seed(seed)
    template_name: str = template_file.replace('/', '--').replace('\\', '--').replace('.txt', '')
    prompt_generator: ConcatPassagesPromptGenerator = ConcatPassagesPromptGenerator(
        template_file, passage_order, use_all_passages, add_section_title
    )

    predictions: List[Dict] = []
    for instance in tqdm(instances):

        prompt_data: Dict = prompt_generator.make_prompt_for_instance(instance)
        output: Dict = llm.prompt_with_special_tokens(prompt_data['prompt'])
        prompt_data['params'] = {
            'split': split, 'llm': llm.get_specs(), 'template': template_file,
            'add_section_title': add_section_title, 'seed': seed
        }
        prompt_data['output'] = output
        predictions.append(prompt_data)

    file_name: str = f'st3-concat__{template_name}__{llm.specs_string()}-s{seed}-t{add_section_title}-O{passage_order}-a{use_all_passages}.{split}.jsonl'
    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    write_jsonl(join(directory, file_name), predictions)


def run_context_wise_experiments(
        llm: LLM, instances: List[Dict], template_file: str, seed: int, split: str
):
    template: str = read_text(join('./prompt_templates', template_file))
    set_seed(seed)
    predictions: List[Dict] = []
    template_name: str = template_file.replace('/', '--').replace('\\', '--').replace('.txt', '')

    for instance in tqdm(instances):
        p0_text: str = instance['argument']['accurate_premise_p0']['premise']

        contexts: List[Dict] = [{
            'text': fallacy['fallacy_context'],
            'passage': fallacy['id']
        }
            for fallacy in instance['argument']['fallacies']
            if len(fallacy['fallacy_context'].strip()) > 0
        ]

        claim: str = instance['argument']['claim']
        for ctx in contexts:
            prompt: str = make_fallacy_prompt_complete(template, claim, p0_text, ctx['text'])
            output: Dict = llm.prompt_with_special_tokens(prompt)
            predictions.append({
                'argument_id': instance['id'],
                'output': output,
                'claim': claim,
                'p0': p0_text,
                'p0_id': 'accurate_premise',
                'context': ctx,
                'params': {
                    'split': split, 'llm': llm.get_specs(), 'template': template_file, 'seed': seed
                }
            })

        # Add one more with only the p0 passage
        prompt: str = make_fallacy_prompt_complete(template, claim, p0_text, '')
        output: Dict = llm.prompt_with_special_tokens(prompt)
        predictions.append({
            'argument_id': instance['id'],
            'output': output,
            'claim': claim,
            'p0': p0_text,
            'p0_id': 'accurate_premise',
            'context': None,
            'params': {
                'split': split, 'llm': llm.get_specs(), 'template': template_file,  'seed': seed
            }
        })

    file_name: str = f'st3-ctx-wise__{template_name}__{llm.specs_string()}-s{seed}.{split}.jsonl'
    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    write_jsonl(join(directory, file_name), predictions)


def run_passagewise_experiments(
        llm: LLM, instances: List[Dict], template_file: str, add_section_title: bool,
        seed: int, split: str
):
    template: str = read_text(join('./prompt_templates', template_file))
    set_seed(seed)
    predictions: List[Dict] = []
    template_name: str = template_file.replace('/', '--').replace('\\', '--').replace('.txt', '')

    for instance in tqdm(instances):
        p0_passages: List[Dict] = get_p0_passages(instance, add_section_title=add_section_title, use_p0_as_backup=True)
        random.shuffle(p0_passages)

        all_passage_keys: Iterable[str] = filter(
            lambda x: x != p0_passages[0]['passage'], get_sorted_passages(instance, from_full_study=False)
        )

        study: Dict = instance['study']['selected_passages']
        all_passage_text: List[Dict] = [{
            'text': get_passage_text(study[passage_key], add_section_title=add_section_title),
            'passage': passage_key
        } for passage_key in all_passage_keys]

        claim: str = instance['argument']['claim']
        p0: str = p0_passages[0]['text']
        for passage in all_passage_text:
            prompt: str = make_fallacy_prompt_complete(template, claim, p0, passage['text'])
            output: Dict = llm.prompt_with_special_tokens(prompt)
            predictions.append({
                'argument_id': instance['id'],
                'output': output,
                'claim': claim,
                'p0': p0,
                'p0_id': p0_passages[0]['passage'],
                'context': passage,
                'params': {
                    'split': split, 'llm': llm.get_specs(), 'template': template_file,
                    'add_section_title': add_section_title, 'seed': seed
                }
            })

        # Add one more with only the p0 passage
        prompt: str = make_fallacy_prompt_complete(template, claim, p0, '')
        output: Dict = llm.prompt_with_special_tokens(prompt)
        predictions.append({
            'argument_id': instance['id'],
            'output': output,
            'claim': claim,
            'p0': p0,
            'p0_id': p0_passages[0]['passage'],
            'context': None,
            'params': {
                'split': split, 'llm': llm.get_specs(), 'template': template_file,
                'add_section_title': add_section_title, 'seed': seed
            }
        })

    file_name: str = f'st3-passage-wise__{template_name}__{llm.specs_string()}-s{seed}-t{add_section_title}.{split}.jsonl'
    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    write_jsonl(join(directory, file_name), predictions)


def agg_claim_level_predictions(prompt_level_predictions: List[Dict]) -> Dict:
    current_from_passages: Set = set()
    fallacies: List[Dict] = []
    for pred in prompt_level_predictions:
        if len(pred['fallacies']) > 0:
            if 'from' in pred:
                current_from_passages |= set(pred['from'])
            fallacies.extend(pred['fallacies'])
    return {
        'fallacies': fallacies,
        'from': list(set(current_from_passages))
    }


def parse_true_claims_prediction_file(file_name: str):
    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    parser: ArgumentReconstructionParser = ArgumentReconstructionParser(file_name, FallacyMapper())
    predictions: List[Dict] = list(read_jsonl(join(directory, file_name)))

    claim_level_predictions: Dict[str, List[Dict]] = defaultdict(list)
    for i, prediction in enumerate(predictions):
        original_instance: Dict = prediction['instance']
        parsed: Dict = parser.parse(prediction)

        instance_id = original_instance['id'] if 'id' in original_instance else original_instance['index']
        instance_arg_id = f'{original_instance["dataset"]}-{instance_id}'

        claim_level_predictions[instance_arg_id].append({
            'fallacies': parsed['predicted_fallacies'],
            'is_parsed': parsed['is_parsed'],
            'prediction_id': i+1,
            'predicted_valid': parsed['valid_logic'],
            'original': original_instance
        })

    for key in claim_level_predictions:
        assert len(claim_level_predictions[key]) == 1

        dest_directory: str = get_prediction_directory('argument-reconstruction')
        write_jsonl(join(dest_directory, file_name), [{
                'argument': arg_id,
                'prompt_based_predictions': claim_level_predictions[arg_id],
                'all_predictions': agg_claim_level_predictions(claim_level_predictions[arg_id])
            } for arg_id in claim_level_predictions.keys()
        ])


def parse_missciplus_prediction_file(file_name: str):

    def extract_passage_keys(pred: Dict) -> List[str]:
        if 'passages' in pred:
            passages: Iterable[str] = map(lambda p: p['passage'], pred['passages'])
            passages = map(lambda x: 'accurate_premise' if x is None else x, passages)
            return list(passages)
        else:
            passages: List[str] = [parsed['p0_id'] or 'accurate_premise']
            if parsed['context'] is not None:
                passages.append(parsed['context']['passage'])
            return passages

    directory: str = get_raw_prompt_prediction_directory('argument-reconstruction')
    parser: ArgumentReconstructionParser = ArgumentReconstructionParser(file_name, FallacyMapper())
    predictions: List[Dict] = list(read_jsonl(join(directory, file_name)))

    argument_level_predictions: Dict[str, List[Dict]] = defaultdict(list)
    for i, prediction in enumerate(predictions):
        arg_id: str = prediction['argument_id']
        parsed: Dict = parser.parse(prediction)

        from_passages: List[str] = extract_passage_keys(prediction)
        for j, f in enumerate(parsed['predicted_fallacies']):
            f['prediction_id'] = f'{i+1}-{j+1}'
            f['from'] = from_passages

        argument_level_predictions[arg_id].append({
            'fallacies': parsed['predicted_fallacies'],
            'from': from_passages,
            'is_parsed': parsed['is_parsed'],
            'prediction_id': i+1,
            'predicted_valid': parsed['valid_logic'],

        })

    cnt = 0
    for k in argument_level_predictions:
        cnt += len(argument_level_predictions[k])
    assert cnt == len(predictions)

    dest_directory: str = get_prediction_directory('argument-reconstruction')
    write_jsonl(join(dest_directory, file_name), [{
            'argument': arg_id,
            'prompt_based_predictions': argument_level_predictions[arg_id],
            'all_predictions': agg_claim_level_predictions(argument_level_predictions[arg_id])
        } for arg_id in argument_level_predictions.keys()
    ])


def evaluate_binary(file: str):
    directory: str = get_prediction_directory('argument-reconstruction')
    parsed_predictions: List[Dict] = list(read_jsonl(join(directory, file)))

    count_fallacious: int = 0
    for pred in parsed_predictions:
        assert len(pred['prompt_based_predictions']) == 1, 'only consider concat-based prompts'
        prompt_based_pred: Dict = pred['prompt_based_predictions'][0]
        is_valid: bool = prompt_based_pred['predicted_valid']
        num_fallacies: int = len(prompt_based_pred['fallacies'])

        if not is_valid and num_fallacies > 0:
            count_fallacious += 1

    metrics: Dict = {
        'instances': len(parsed_predictions),
        'predicted-fallacious': count_fallacious / len(parsed_predictions)
    }
    out_file = 'binary-eval__' + file.replace('.jsonl.jsonl', '.jsonl').replace('.jsonl', '.eval-binary.json')
    write_json(metrics, dest=join(directory, out_file), pretty=True)


def map_prediction_file(file_name: str, instances: List[Dict], judge: SFTLlama3Judge, overwrite: bool):
    directory: str = get_prediction_directory('argument-reconstruction')
    file_name_dest: str = file_name.replace('.jsonl', '.mapped.jsonl')
    if not overwrite and exists(join(directory, file_name_dest)):
        print('SKIP:', file_name)
        return

    argument_fallacy_mapper: ArgumentFallacyMapper = ArgumentFallacyMapper(judge)
    id_to_gold: Dict[str, Dict] = {inst['id']: inst for inst in instances}
    id_to_pred: Dict[str, Dict] = {pred['argument']: pred for pred in read_jsonl(join(directory, file_name))}
    mapped_predictions: List[Dict] = []
    for key in sorted(list(id_to_gold.keys())):
        pred: Dict = id_to_pred[key]
        gold: Dict = id_to_gold[key]
        mapped_predictions.append(argument_fallacy_mapper.map(pred, gold))

    write_jsonl(join(directory, file_name_dest), mapped_predictions)


def evaluate_prediction_file(file: str, split: str, use_all_passages: bool, use_only_mapped_context: bool):
    evaluator: GenClassifyEvaluatorMissciPlus = GenClassifyEvaluatorMissciPlus(split=split)
    metrics = evaluator.evaluate_file(file, use_all_passages=use_all_passages, use_only_mapped_context=use_only_mapped_context)
    print(file)
    print(metrics['all']['claim_level_multi_label']['f1_micro'])
    print()


def extract_topk_alignment(file: str, k: Union[str, int], split: str, mapped_only: bool):
    evaluator: GenClassifyEvaluatorMissciPlus = GenClassifyEvaluatorMissciPlus(split=split)
    prediction_dict: Dict = evaluator.make_prediction_dict(file, k=k, mapped_only=mapped_only)

    directory: str = get_prediction_directory('argument-reconstruction')
    write_json(prediction_dict, join(directory, file.replace('jsonl', f'.align-k-{k}.json')), pretty=True)


def make_llm(args: Dict) -> LLM:
    llm_key: str = args['<llm>']
    if llm_key == 'llama2':
        llm: LLM = BasicLlama2(
            llama_size='70b',
            max_new_token_len=3000,
            temperature=float(args['--temperature']) if args['--temperature'] is not None else None
        )
    elif llm_key == 'llama3':
        if args['--small']:
            llm: LLM = BasicLlama3Pipeline(
                llama_size='8b',
                run_8bit=False,
                max_new_token_len=3000,
                temperature=float(args['--temperature']) if args['--temperature'] is not None else None
            )
        else:
            llm: LLM = BasicLlama3Pipeline(
                llama_size='70b',
                max_new_token_len=3000,
                temperature=float(args['--temperature']) if args['--temperature'] is not None else None
            )
    elif llm_key == 'chatgpt':
        from missci.modeling.basic_llm.basic_chatgpt import BasicAnyGPT
        llm: LLM = BasicAnyGPT(
            gpt_version='gpt-35-turbo-16k-no-filter',
            max_new_token_len=3000
        )
    elif llm_key == 'gpt4t':
        from missci.modeling.basic_llm.basic_chatgpt import BasicAnyGPT
        llm: LLM = BasicAnyGPT(
            gpt_version='gpt4-turbo-128k',
            max_new_token_len=3000,
            llm_name="gpt4-turbo"
        )
    else:
        raise NotImplementedError(llm_key)
    return llm


def main():
    args = docopt(__doc__)

    split = 'dev' if args['--dev'] else args['--file'] or 'test'
    if split in {'dev', 'test'}:
        instances: List[Dict] = MappedDataLoader().load_raw_arguments(split)
    else:
        instances = []

    if args['passage-wise']:
        assert len(instances) > 0
        run_passagewise_experiments(
            llm=make_llm(args),
            instances=instances,
            template_file=args['<template>'],
            add_section_title=args['--add-section'],
            seed=int(args['<seed>']),
            split=split
        )
    elif args['concat-passages-correct']:
        assert len(instances) > 0
        run_concat_passages_true_experiments(
            llm=make_llm(args),
            file_name=args['<file>'],
            template_file=args['<template>'],
            seed=int(args['<seed>'])
        )
    elif args['context-wise']:
        assert len(instances) > 0
        run_context_wise_experiments(
            llm=make_llm(args),
            instances=instances,
            template_file=args['<template>'],
            seed=int(args['<seed>']),
            split=split
        )
    elif args['concat-passages']:
        assert len(instances) > 0
        run_concat_passages_experiments(
            llm=make_llm(args),
            instances=instances,
            template_file=args['<template>'],
            add_section_title=args['--add-section'],
            seed=int(args['<seed>']),
            split=split,
            passage_order=args['<order>'].strip(),
            use_all_passages=args['--all']
        )
    elif args['concat-context']:
        assert len(instances) > 0
        run_concat_context_experiments(
            llm=make_llm(args),
            instances=instances,
            template_file=args['<template>'],
            seed=int(args['<seed>']),
            split=split
        )

    elif args['parse']:
        file: str = args['<file>']
        if split in {'test', 'dev'}:
            parse_missciplus_prediction_file(file)
        else:
            parse_true_claims_prediction_file(file)
    elif args['evaluate-binary']:
        evaluate_binary(args['<file>'])
    elif args['map']:
        assert len(instances) > 0
        file: str = args['<file>']

        model_name: str = 'full-sft__sft-llama3-8b-same-reasoning-4_ep5_ba1_sc-linear_lr00005_a16_drp02_r64'
        judge: SFTLlama3Judge = SFTLlama3Judge.load(
            model_name=model_name,
            instructions=get_judge_instruction('same-reasoning-4'),
            run_8bit=True
        )
        map_prediction_file(file, instances, judge, args['--overwrite'])

    elif args['evaluate']:
        assert len(instances) > 0
        file: str = args['<file>']
        evaluate_prediction_file(file, split, args['--all'], args['--only-mapped-context'])
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()

