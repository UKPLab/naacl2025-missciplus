from collections import defaultdict
from os.path import join
from typing import Dict, List

import datasets
import pandas as pd
from datasets import DatasetDict, Dataset, Features

from missci.data.load_scifact import get_scifact_id2label


def records_to_dataset(records: List[Dict], features: Features) -> Dataset:
    data_dict: Dict[str, List] = defaultdict(list)
    for record in records:
        for key in ['id', 'claim', 'label', 'evidence_full_passage']:
            data_dict[key].append(record[key])
    return Dataset.from_dict(data_dict, features=features)


def get_covid_fact_id2label() -> Dict[int, str]:
    return {
        0: 'SUPPORT',
        1: 'CONTRADICT'
    }


def get_covidfact_claims(label: bool, split: str, data_directory: str) -> List[Dict]:
    if split == 'test':
        split = 'test1'
    df = make_df(data_directory, split, {'entailment': True, 'not_entailment': False}).rename(columns={
        'Sentence2': 'claim',
        'sentence1': 'evidence'
    })
    df['dataset'] = 'covidfact'
    df['split'] = split
    return df.loc[df['label'] == label, ['index', 'claim', 'evidence', 'label', 'dataset', 'split']].to_dict('records')


def make_df(directory: str, split: str, conversion_dict: Dict):
    df: pd.DataFrame = pd.read_csv(join(directory, f'{split}.tsv'), sep='\t')
    if split != 'dev':
        df.loc[:, 'label'] = df['label'].apply(lambda x: conversion_dict[x])
    else:
        df['label'] = df['entailment'].apply(lambda x: conversion_dict[x])
    return df


def get_covid_fact_label2id() -> Dict[str, int]:
    id2label: Dict[int, str] = get_covid_fact_id2label()
    return {
        id2label[k]: k for k in id2label
    }


def get_claim_passage_covid_fact(directory: str, three_labels: bool = False) -> DatasetDict:
    if three_labels:
        # Only needed when trained together with other datasets
        names: List[str] = list(map(lambda x: get_scifact_id2label()[x], range(3)))
        num_labels: int = 3
    else:
        names: List[str] = list(map(lambda x: get_covid_fact_id2label()[x], range(2)))
        num_labels: int = 2
    dataset_features: Features = Features({
        'id': datasets.Value(dtype='int32', id=None),
        'claim': datasets.Value(dtype='string', id=None),
        'label': datasets.ClassLabel(num_classes=num_labels, names=names),
        'evidence_full_passage': datasets.Value(dtype='string', id=None)
    })

    conversion_dict: Dict[str, str] = {'entailment': 'SUPPORT', 'not_entailment': 'CONTRADICT'}
    data_subsets: Dict[str, Dataset] = dict()

    for split in ['train', 'dev', 'test1']:
        df = make_df(directory, split, conversion_dict)
        # Just so that it is the same format as SciFact
        df = df.rename(columns={
            'sentence1': 'evidence_full_passage',
            'Sentence2': 'claim',
            'index': 'id'
        })
        instances: List[Dict] = df.to_dict('records')

        if split == 'train':
            split_name: str = 'train'
        elif split == 'dev':
            split_name = 'validation'
        elif split == 'test1':
            split_name = 'test'
        else:
            raise ValueError(split)

        data_subsets[split_name] = records_to_dataset(instances, dataset_features)

    return DatasetDict(data_subsets)
