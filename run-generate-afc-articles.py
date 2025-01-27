"""run-generate-afc-articles.py

Usage:
  run-generate-afc-articles.py llama <template> <seed> [--temperature=<temperature>] [--dev] [--llm-spec=<spec>]
  run-generate-afc-articles.py chatgpt <template> <seed> [--dev]
  run-generate-afc-articles.py evidence llama <template> <seed> [--temperature=<temperature>] [--dev] [--llm-spec=<spec>]
  run-generate-afc-articles.py evidence chatgpt <template> <seed> [--temperature=<temperature>] [--dev]
  run-generate-afc-articles.py parse <file> <template-name> [--dev]
  run-generate-afc-articles.py true-claims extract <num>
  run-generate-afc-articles.py true-claims extract-scifact <num>
  run-generate-afc-articles.py true-claims llama <template> <seed> <file> [--temperature=<temperature>] [--llm-spec=<spec>]
  run-generate-afc-articles.py true-claims chatgpt <template> <seed> <file> [--temperature=<temperature>]
  run-generate-afc-articles.py true-claims-evidence llama <template> <seed> <file> [--temperature=<temperature>] [--llm-spec=<spec>]
  run-generate-afc-articles.py true-claims-evidence chatgpt <template> <seed> <file> [--temperature=<temperature>]

"""
import os.path
import random
from collections import Counter
from os import makedirs
from os.path import join, exists
from typing import Dict, List, Optional, Iterable, Tuple

from datasets import Dataset
from docopt import docopt
from tqdm import tqdm
from transformers import set_seed

from missci.data.load_covidfact import get_covidfact_claims
from missci.data.load_healthver import get_healthver_claims
from missci.data.load_scifact import get_claim_passage_scifact, get_scifact_label2id
from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.modeling.basic_llm.basic_chatgpt import BasicAnyGPT
from missci.modeling.basic_llm.basic_llama2 import BasicLlama2
from missci.modeling.basic_llm.basic_llama3 import BasicLlama3Pipeline
from missci.modeling.basic_llm.llm import LLM
from missci.modeling.prompting import filled_template_to_prompt_gpt
from missci.output_parser.llm_output_parser_afc_article import AFCArticleParser
from missci.util.fileutil import write_jsonl, read_text, read_jsonl, write_json

PREDICTION_DIRECTORY_AFC: str = './predictions/afc-articles'


def make_afc_prompt(claim: str, template: str) -> str:
    loaded_template: str = read_text(join('./prompt_templates', template))
    assert '@@claim@@' in loaded_template
    return loaded_template.replace('@@claim@@', claim)


def make_stance_prompt_for_true_claims(claim: str, evidence: str, template: str):
    loaded_template: str = read_text(join('./prompt_templates', template))
    assert '@@claim@@' in loaded_template
    assert '@@evidence@@' in loaded_template
    return loaded_template.replace('@@claim@@', claim).replace('@@evidence@@', evidence)


def make_stance_prompt(instance: Dict, template: str) -> str:

    def to_passage_string(passage: Dict) -> str:
        passage_num = passage['passage_id'].split('-')[-1]
        title = f'Passage {passage_num}'
        if passage['section'] != '':
            title += f' (Section: "{passage["section"]}")'
        title += '\n'
        return title + ' '.join(passage['sentences'])

    claim: str = instance['argument']['claim']
    loaded_template: str = read_text(join('./prompt_templates', template))
    assert '@@claim@@' in loaded_template
    assert '@@evidence@@' in loaded_template

    all_mappings: List[Dict] = instance['argument']['accurate_premise_p0']['mapping']
    for fallacy in instance['argument']['fallacies']:
        all_mappings.extend(fallacy['mapping'])
    all_mappings = list(set(map(lambda m: m['passage'], all_mappings)))
    all_mappings = sorted(all_mappings, key=lambda x: int(x.split('-')[-1]))
    passages: Iterable[Dict] = map(lambda passage_key: instance['study']['selected_passages'][passage_key], all_mappings)
    passages: str = '\n\n'.join(list(map(to_passage_string, passages)))

    return loaded_template.replace('@@claim@@', claim).replace('@@evidence@@', passages)


def generate_stance(
        template: str, split: str, instances: List[Dict], llm_type: str, temperature: Optional[float], seed: int,
        llm: LLM
):

    set_seed(seed)
    _, file = os.path.split(template)
    template_name: str = file.replace('.txt', '')
    output_name: str = f'evidence_afc_{llm.specs_string()}__{template_name}__s-{seed}.{split}.jsonl'

    predictions: List[Dict] = []
    for instance in tqdm(instances):

        out: Dict = llm.prompt_with_special_tokens(make_stance_prompt(instance, template))
        claim: str = instance['argument']['claim']
        arg_id: str = instance['id']
        out['claim'] = claim
        out['argument'] = arg_id
        predictions.append(out)

    if not exists(PREDICTION_DIRECTORY_AFC):
        makedirs(PREDICTION_DIRECTORY_AFC)
    write_jsonl(join(PREDICTION_DIRECTORY_AFC, output_name), predictions)


def generate_stance_for_true_claims(
        template: str, file_name: str, llm_type: str, temperature: Optional[float], seed: int,
        llm: LLM
):

    set_seed(seed)
    _, file = os.path.split(template)
    template_name: str = file.replace('.txt', '')
    file_name_base = file_name.replace('.jsonl', '')
    output_name: str = f'{file_name_base}_{llm.specs_string()}__{template_name}__s-{seed}.jsonl'

    predictions: List[Dict] = []
    for instance in tqdm(list(read_jsonl(join('./afc_data', file_name)))):
        out: Dict = llm.prompt_with_special_tokens(make_stance_prompt_for_true_claims(
                instance['claim'], instance['evidence'], template
            ))
        out['instance'] = instance
        predictions.append(out)

    if not exists(PREDICTION_DIRECTORY_AFC):
        makedirs(PREDICTION_DIRECTORY_AFC)
    write_jsonl(join(PREDICTION_DIRECTORY_AFC, output_name), predictions)


def generate_afc_articles_for_true_claims(
    template: str, file_name: str, llm: LLM, seed: int
):
    set_seed(seed)
    _, file = os.path.split(template)
    template_name: str = file.replace('.txt', '')
    file_name_base = file_name.replace('.jsonl', '')
    output_name: str = f'{file_name_base}__{llm.specs_string()}__{template_name}__s-{seed}.jsonl'

    predictions: List[Dict] = []
    for instance in tqdm(list(read_jsonl(join('./afc_data', file_name)))):

        out: Dict = llm.prompt_with_special_tokens(make_afc_prompt(instance['claim'], template))
        out['instance'] = instance
        predictions.append(out)

    if not exists(PREDICTION_DIRECTORY_AFC):
        makedirs(PREDICTION_DIRECTORY_AFC)
    write_jsonl(join(PREDICTION_DIRECTORY_AFC, output_name), predictions)


def generate_afc_articles_with_llama(
        template: str, split: str, instances: List[Dict], temperature: Optional[float], seed: int,
        llm: LLM
):

    set_seed(seed)

    _, file = os.path.split(template)
    template_name: str = file.replace('.txt', '')
    output_name: str = f'{llm.specs_string()}__{template_name}__s{seed}.{split}.jsonl'

    predictions: List[Dict] = []
    for instance in tqdm(instances):
        claim: str = instance['argument']['claim']
        out: Dict = llm.prompt_with_special_tokens(make_afc_prompt(claim, template))
        arg_id: str = instance['id']
        out['claim'] = claim
        out['argument'] = arg_id
        predictions.append(out)

    if not exists(PREDICTION_DIRECTORY_AFC):
        makedirs(PREDICTION_DIRECTORY_AFC)
    write_jsonl(join(PREDICTION_DIRECTORY_AFC, output_name), predictions)


def generate_afc_articles_with_chatgpt(
        template: str, split: str, instances: List[Dict],seed: int
):
    set_seed(seed)
    llm: LLM = BasicAnyGPT(max_new_token_len=3000)

    _, file = os.path.split(template)
    template_name: str = file.replace('.txt', '')
    output_name: str = f'chatgpt__{template_name}__s-{seed}__.{split}.jsonl'

    predictions: List[Dict] = []
    for instance in tqdm(instances):
        claim: str = instance['argument']['claim']
        prompt: str = filled_template_to_prompt_gpt(make_afc_prompt(claim, template))
        out: Dict = llm.prompt(prompt)
        arg_id: str = instance['id']
        out['claim'] = claim
        out['argument'] = arg_id
        predictions.append(out)

    if not exists(PREDICTION_DIRECTORY_AFC):
        makedirs(PREDICTION_DIRECTORY_AFC)
    write_jsonl(join(PREDICTION_DIRECTORY_AFC, output_name), predictions)


def get_afc_verdict_labels(template_name: str) -> Tuple[List[str], str]:
    if not template_name.endswith('.txt'):
        template_name += '.txt'

    template_name_to_labels: Dict[str, List[str]] = {
        'fc-stance-evidence-binary.txt': ['Correct', 'Incorrect'],
        'fc-stance-evidence-ternary.txt': ['Correct', 'Incorrect', 'Not Enough Information'],
        'fc_article_binary.txt': ['True', 'False'],
        'fc_article_fine-grained.txt': ['True', 'Mostly True', 'Mixed', 'Mostly False', 'False'],
        'fc_article_ternary-mixed.txt': ['True', 'Mixed', 'False'],
        'fc_article_ternary-nei.txt': ['True', 'Unknown', 'False'],
        'knowledge_binary.txt': ['True', 'False'],
        'knowledge_ternary-nei.txt': ['True', 'Unknown', 'False'],
        'knowledge_elaborate_binary.txt': ['True', 'False'],
        'knowledge_elaborate_ternary.txt': ['True', 'Unknown', 'False']
    }

    template_name_to_verdict_key = {
        'fc-stance-evidence-binary.txt': 'verdict',
        'fc-stance-evidence-ternary.txt': 'verdict',
        'fc_article_binary.txt': 'verdict',
        'fc_article_fine-grained.txt': 'verdict',
        'fc_article_ternary-mixed.txt': 'verdict',
        'fc_article_ternary-nei.txt': 'verdict',
        'knowledge_binary.txt': 'veracity',
        'knowledge_elaborate_binary.txt': 'veracity',
        'knowledge_ternary-nei.txt': 'veracity',
        'knowledge_elaborate_ternary.txt': 'veracity'
    }

    labels = template_name_to_labels[template_name]
    verdict_key: str = template_name_to_verdict_key[template_name]
    return list(map(lambda lbl: lbl.lower(), labels)), verdict_key


def parse_llm_output(file: str, label_key: str):

    labels, verdict_key = get_afc_verdict_labels(label_key)
    parser: AFCArticleParser = AFCArticleParser(labels, verdict_key)

    verdicts: List = []

    predictions: List[Dict] = list(read_jsonl(join(PREDICTION_DIRECTORY_AFC, file)))
    for prediction in tqdm(predictions):
        output: str = prediction['output'] if 'output' in prediction else prediction['answer']
        verdict: Optional[str] = parser.parse(output)
        prediction['veracity'] = verdict
        verdicts.append(verdict)

    write_jsonl(join(PREDICTION_DIRECTORY_AFC, file.replace('.jsonl', '.parsed.jsonl')), predictions)
    count_other: int = 0

    result = {}

    for verdict, count in Counter(verdicts).most_common():
        print(f'{count} --> {verdict} ({round(100 * count / len(predictions), 1)} %)')
        if verdict not in labels:
            count_other += count
        else:
            result[verdict] = count

    print(f'Total number of "other" verdicts: {count_other} ({round(100 * count_other / len(predictions), 1)} %)')
    result['other'] = count_other
    result['num_predictions'] = len(predictions)

    num_no_verdict: int = len(list(filter(lambda v: v is None, verdicts)))
    print(f'Total number of NO verdicts: {num_no_verdict} ({round(100 * num_no_verdict / len(predictions), 1)} %)')
    result['no-verdict'] = num_no_verdict

    write_json(result, join(PREDICTION_DIRECTORY_AFC, file.replace('.jsonl', '.metrics.json')), pretty=True)


def extract_true_claims_scifact(num_claims_per_source: int) -> None:
    dataset: Dataset = get_claim_passage_scifact()['test']
    lbl2id: Dict = get_scifact_label2id()

    df = dataset.to_pandas()
    df = df[df['label'] == lbl2id['SUPPORT']].drop_duplicates(subset='id').sample(frac=1., random_state=1)
    samples: List[Dict] = df.to_dict('records')[:num_claims_per_source]

    def map_sample(sample):
        return {
            'id': sample['id'],
            'claim': sample['claim'],
            'evidence': sample['title'] + '\n' + sample['evidence_full_passage'],
            'label': True,
            'dataset': 'scifact',
            'split': 'test'
        }
    assert len(samples) == num_claims_per_source
    write_jsonl(join('./afc_data', f'scifact-true-{num_claims_per_source}.jsonl'), list(map(map_sample, samples)))


def extract_true_claims(num_claims_per_source: int) -> None:
    covidfact_claims: List[Dict] = get_covidfact_claims(
        label=True, split='test', data_directory='./afc_data/RTE-covidfact'
    )
    assert len(covidfact_claims) >= num_claims_per_source

    healthver_claims: List[Dict] = get_healthver_claims(label='SUPPORT', split='test', data_directory='./afc_data/healthver')
    assert len(healthver_claims) >= num_claims_per_source

    random.seed(1)
    random.shuffle(covidfact_claims)
    random.shuffle(healthver_claims)
    true_claims: List[Dict] = covidfact_claims[:50] + healthver_claims[:50]
    write_jsonl(join('./afc_data', f'cov-health-true-{num_claims_per_source}.jsonl'), true_claims)


def get_llama_from_specs(args: Dict) -> LLM:
    llm_spec = args['--llm-spec']
    temperature: Optional[float] = float(args['--temperature']) if args['--temperature'] is not None else None
    if llm_spec is None:
        llm: LLM = BasicLlama2(llama_size='70b', run_8bit=True, temperature=temperature)
    elif llm_spec == 'llama3-8b':
        llm = BasicLlama3Pipeline('8b', run_8bit=False, temperature=temperature)
    elif llm_spec == 'llama3-70b':
        llm = BasicLlama3Pipeline('70b', run_8bit=True, temperature=temperature)
    elif llm_spec == 'llama2-70b':
        llm = BasicLlama2('70b', run_8bit=True, temperature=temperature)
    else:
        raise NotImplementedError(llm_spec)
    return llm


def main():
    args = docopt(__doc__)
    split: str = 'dev' if args['--dev'] else 'test'
    instances: List[Dict] = MappedDataLoader().load_raw_arguments(split)

    if args['evidence']:
        if args['llama']:
            llm: LLM = get_llama_from_specs(args)
            temperature: Optional[float] = float(args['--temperature']) if args['--temperature'] is not None else None
            generate_stance(
                args['<template>'], split, instances, 'llama',
                temperature,
                int(args['<seed>']),
                llm
            )
        elif args['chatgpt']:
            llm: LLM = BasicAnyGPT(max_new_token_len=1000)
            generate_stance(
                args['<template>'], split, instances, 'chatgpt',
                float(args['--temperature']) if args['--temperature'] is not None else None,
                int(args['<seed>']),
                llm
            )
        else:
            raise NotImplementedError()

    elif args['true-claims']:
        if args['extract']:
            num: int = int(args['<num>'])
            extract_true_claims(num)
        elif args['extract-scifact']:
            num: int = int(args['<num>'])
            extract_true_claims_scifact(num)
        elif args['llama']:
            llm: LLM = get_llama_from_specs(args)
            generate_afc_articles_for_true_claims(
                args['<template>'], args['<file>'], llm, int(args['<seed>'])
            )
        elif args['chatgpt']:
            generate_afc_articles_for_true_claims(
                args['<template>'], args['<file>'], BasicAnyGPT(max_new_token_len=3000), int(args['<seed>'])
            )
    elif args['true-claims-evidence']:
        if args['llama']:
            llm: LLM = get_llama_from_specs(args)
            temperature: Optional[float] = float(args['--temperature']) if args['--temperature'] is not None else None
            generate_stance_for_true_claims(
                args['<template>'], args['<file>'], 'llama', temperature,  int(args['<seed>']), llm
            )
        elif args['chatgpt']:
            llm: LLM = BasicAnyGPT(max_new_token_len=3000)
            temperature: Optional[float] = float(args['--temperature']) if args['--temperature'] is not None else None
            generate_stance_for_true_claims(
                args['<template>'], args['<file>'], 'chatgpt', temperature,  int(args['<seed>']), llm
            )

    elif args['llama']:

        llm: LLM = get_llama_from_specs(args)
        temperature: Optional[float] = float(args['--temperature']) if args['--temperature'] is not None else None

        generate_afc_articles_with_llama(
            args['<template>'], split, instances,
            temperature,
            int(args['<seed>']),
            llm
        )

    elif args['parse']:
        parse_llm_output(args['<file>'], args['<template-name>'])

    elif args['chatgpt']:
        generate_afc_articles_with_chatgpt(
            args['<template>'], split, instances,
            int(args['<seed>'])
        )
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
