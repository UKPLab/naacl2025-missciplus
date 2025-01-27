from collections import defaultdict
from typing import Dict, Optional, List

import datasets
from datasets import DatasetDict, Features, Dataset

from missci.data.load_covidfact import get_claim_passage_covid_fact, get_covid_fact_id2label
from missci.data.load_healthver import get_claim_passage_healthver, get_healthver_id2label
from missci.data.load_scifact import get_claim_passage_scifact, get_scifact_id2label


def get_dataset_and_int2lbl(task_name: str, seed: Optional[int] = None):
    if task_name == 'scifact':
        dataset: DatasetDict = get_claim_passage_scifact()
        int2lbl: Dict[int, str] = get_scifact_id2label()
    elif task_name == 'healthver':
        dataset: DatasetDict = get_claim_passage_healthver('./afc_data/healthver')
        int2lbl: Dict[int, str] = get_healthver_id2label()
    elif task_name == 'covidfact':
        dataset: DatasetDict = get_claim_passage_covid_fact('./afc_data/RTE-covidfact')
        int2lbl: Dict[int, str] = get_covid_fact_id2label()
    elif task_name == 'covidfact-3label':
        dataset: DatasetDict = get_claim_passage_covid_fact('./afc_data/RTE-covidfact')
        int2lbl: Dict[int, str] = get_covid_fact_id2label()
    elif task_name == 'sci-health-cov':
        int2lbl: Dict[int, str] = get_scifact_id2label()
        assert seed is not None
        dataset: DatasetDict = get_claim_passage_scifact_healthver_covidfact(seed=seed)
    else:
        raise ValueError(task_name)
    return dataset, int2lbl


def get_claim_passage_scifact_healthver_covidfact(seed: int) -> DatasetDict:
    dataset_scifact, int2lbl = get_dataset_and_int2lbl('scifact')
    dataset_healthver, _ = get_dataset_and_int2lbl('healthver')
    dataset_covid_fact, _ = get_dataset_and_int2lbl('covidfact-3label')
    names: List[str] = list(map(lambda x: int2lbl[x], range(3)))
    dataset_features: Features = Features({
        'id': datasets.Value(dtype='int32', id=None),
        'task': datasets.Value(dtype='string', id=None),
        'claim': datasets.Value(dtype='string', id=None),
        'label': datasets.ClassLabel(num_classes=3, names=names),
        'evidence_full_passage': datasets.Value(dtype='string', id=None)
    })

    subsets: Dict[str, Dataset] = dict()
    for split in ['train', 'validation', 'test']:
        data_dict_for_split: Dict[str, List] = defaultdict(list)
        for task, dataset in [
            ('scifact', dataset_scifact), ('healthver', dataset_healthver), ('covidfact', dataset_covid_fact)
        ]:
            for entry in dataset[split]:
                data_dict_for_split['id'].append(entry['id'])
                data_dict_for_split['task'].append(task)
                data_dict_for_split['claim'].append(entry['claim'])
                data_dict_for_split['label'].append(entry['label'])
                data_dict_for_split['evidence_full_passage'].append(entry['evidence_full_passage'])

        subsets[split] = Dataset.from_dict(data_dict_for_split, features=dataset_features).shuffle(seed=seed)

    return DatasetDict(subsets)
