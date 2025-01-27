from collections import defaultdict
from os.path import join
from typing import Dict, List


import datasets
import pandas as pd
from datasets import DatasetDict, Dataset, Features


def records_to_dataset(records: List[Dict], features: Features) -> Dataset:
    data_dict: Dict[str, List] = defaultdict(list)
    for record in records:
        for key in ['id', 'claim', 'label', 'evidence_full_passage']:
            data_dict[key].append(record[key])
    return Dataset.from_dict(data_dict, features=features)


def get_healthver_id2label() -> Dict[int, str]:
    return {
        0: 'SUPPORT',
        1: 'NOT_ENOUGH_INFO',
        2: 'CONTRADICT'
    }


def get_healthver_label2id() -> Dict[str, int]:
    id2label: Dict[int, str] = get_healthver_id2label()
    return {
        id2label[k]: k for k in id2label
    }


def get_healthver_claims(label: str, split: str, data_directory: str) -> List[Dict]:
    df: pd.DataFrame = pd.read_csv(join(data_directory, f'healthver_{split}.csv'))
    conversion_dict: Dict[str, str] = {'Neutral': 'NOT_ENOUGH_INFO', 'Supports': 'SUPPORT', 'Refutes': 'CONTRADICT'}
    df.loc[:, 'label'] = df['label'].apply(lambda x: conversion_dict[x])
    df['dataset'] = 'healthver'
    df['split'] = split
    return df.loc[df['label'] == label, ['id', 'claim', 'evidence', 'label', 'dataset', 'split']].to_dict('records')


def get_claim_passage_healthver(directory: str) -> DatasetDict:
    names: List[str] = list(map(lambda x: get_healthver_id2label()[x], range(3)))
    dataset_features: Features = Features({
        'id': datasets.Value(dtype='int32', id=None),
        'claim': datasets.Value(dtype='string', id=None),
        'label': datasets.ClassLabel(num_classes=3, names=names),
        'evidence_full_passage': datasets.Value(dtype='string', id=None)
    })

    conversion_dict: Dict[str, str] = {'Neutral': 'NOT_ENOUGH_INFO', 'Supports': 'SUPPORT', 'Refutes': 'CONTRADICT'}
    data_subsets: Dict[str, Dataset] = dict()
    for split in ['train', 'dev', 'test']:
        df: pd.DataFrame = pd.read_csv(join(directory, f'healthver_{split}.csv'))
        df.loc[:, 'label'] = df['label'].apply(lambda x: conversion_dict[x])

        # Just so that it is the same format as SciFact
        df = df.rename(columns={
            'evidence': 'evidence_full_passage'
        })
        instances: List[Dict] = df.to_dict('records')

        split_name: str = 'validation' if split == 'dev' else split
        data_subsets[split_name] = records_to_dataset(instances, dataset_features)

    return DatasetDict(data_subsets)
