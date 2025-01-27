import random
from os.path import join
from typing import Set, Iterable, List, Dict, Tuple

import pandas as pd
from datasets import DatasetDict, Dataset


def get_data_junks(file: str, shuffle_seed: int = 1, num_folds: int = 5, directory: str = './dataset') -> List[List[Dict]]:

    df = pd.read_csv(join(directory, file))
    df['argument_id'] = df['fallacy_id'].apply(lambda fid: ':'.join(fid.split(':')[:-1]))
    df.loc[df['context'].isna(), 'context'] = df.loc[df['context'].isna(), 'p0']
    df = df.dropna(subset='generated_premise')

    argument_ids: List[str] = sorted(list(set(df['argument_id'])))
    random.seed(shuffle_seed)
    random.shuffle(argument_ids)

    fold_size: int = len(argument_ids) // num_folds
    folds = []
    for i in range(num_folds - 1):
        fold = argument_ids[i * fold_size: (i + 1) * fold_size]
        folds.append(fold)

    # Add the remaining data to the last fold
    folds.append(argument_ids[(num_folds - 1) * fold_size:])

    # Verify
    all_arg_ids: Set[str] = set()
    for fold in folds:
        assert set(fold) & all_arg_ids == set()
        all_arg_ids |= set(fold)
    assert len(all_arg_ids) == len(argument_ids)

    final_folds: Iterable = map(lambda f: df[df['argument_id'].isin(f)], folds)
    cols = [
        'argument_id', 'fallacy_id', 'generated_premise', 'reference_premise', 'predicted_fallacy', 'q4', 'nli_s'
    ]
    for v in [
        'data_id', 'p0', 'context', 'claim'
    ]:
        if v in df.columns:
            cols.append(v)
    final_folds: Iterable = map(lambda f_df: f_df.loc[:, cols].rename(columns={'q4': 'label'}).to_dict('records'), final_folds)
    return list(final_folds)


def get_xval_train_test(junks: List[List[Dict]]) -> Iterable[DatasetDict]:
    for i in range(len(junks)):
        test_junk: List[Dict] = junks[i]
        others: List[List[Dict]] = [
            junk for j, junk in enumerate(junks)
            if j != i
        ]
        assert len(others) == len(junks) - 1
        train_junk: List[Dict] = [
            sample for junk in others for sample in junk
        ]

        assert len(test_junk) + len(train_junk) == sum([len(j) for j in junks])
        yield DatasetDict({
            'train': Dataset.from_list(train_junk), 'test': Dataset.from_list(test_junk)
        })
