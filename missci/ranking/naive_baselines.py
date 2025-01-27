from os.path import join
from typing import List, Dict

from missci.util.fileutil import write_jsonl
from missci.util.passage_util import get_sorted_passages
import random


def run_ordered_baseline(dest_dir: str, instances: List[Dict], use_full_study: bool, prefix: str):
    predictions: List[Dict] = []
    for instance in instances:
        passages = get_sorted_passages(instance, from_full_study=use_full_study)
        predictions.append({
            'id': instance['id'],
            'ranked_passages': passages
        })

    file_name: str = f'baseline_sorted_by_position.jsonl'
    if use_full_study:
        file_name = prefix + file_name.replace('.jsonl', '.fullstudy.jsonl')
    write_jsonl(join(dest_dir, file_name), predictions)
    return file_name


def run_random_baseline(dest_dir: str, instances: List[Dict], seed: int, use_full_study: bool, prefix: str):
    print('Run random baseline with seed:', seed)
    random.seed(seed)

    predictions: List[Dict] = []
    for instance in instances:
        passages = get_sorted_passages(instance, from_full_study=use_full_study)
        random.shuffle(passages)
        predictions.append({
            'id': instance['id'],
            'ranked_passages': passages
        })

    file_name: str = f'baseline_random_seed-{seed}.jsonl'
    if use_full_study:
        file_name = prefix + file_name.replace('.jsonl', '.fullstudy.jsonl')
    write_jsonl(join(dest_dir, file_name), predictions)
    return file_name

