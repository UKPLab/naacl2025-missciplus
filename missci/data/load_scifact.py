from collections import defaultdict
from typing import Dict, List

import random

import datasets
from datasets import DatasetDict, load_dataset, Dataset, Features


def records_to_dataset(records: List[Dict], features: Features) -> Dataset:
    data_dict: Dict[str, List] = defaultdict(list)
    for record in records:
        for key in record:
            data_dict[key].append(record[key])
    return Dataset.from_dict(data_dict, features=features)


def to_corpus_dict(dataset: Dataset) -> Dict[int, Dict]:
    return {
        instance['doc_id']: instance for instance in dataset
    }


def get_scifact_id2label() -> Dict[int, str]:
    return {
        0: 'SUPPORT',
        1: 'NOT_ENOUGH_INFO',
        2: 'CONTRADICT'
    }


def get_scifact_label2id() -> Dict[str, int]:
    id2label: Dict[int, str] = get_scifact_id2label()
    return {
        id2label[k]: k for k in id2label
    }


def get_claim_passage_scifact() -> DatasetDict:
    scifact_corpus: DatasetDict = load_dataset('allenai/scifact', 'corpus')
    scifact_claims: DatasetDict = load_dataset('allenai/scifact', 'claims')

    # Has only the train subset.
    corpus_dict: Dict[int, Dict] = to_corpus_dict(scifact_corpus['train'])

    def to_claim_passage_instance(instance: Dict) -> Dict:
        assert len(instance['cited_doc_ids']) > 0

        # sometimes more documents are linked. Take the relevant one (or the first)
        doc_id: int = int(instance['evidence_doc_id']) if instance['evidence_doc_id'] != '' else instance['cited_doc_ids'][0]

        instance['title'] = corpus_dict[doc_id]['title']
        instance['evidence_text'] = list(map(lambda x: x.strip(), corpus_dict[doc_id]['abstract']))
        instance['evidence_full_passage'] = ' '.join(instance['evidence_text'])

        instance['label'] = instance['evidence_label']
        if instance['label'] == '':
            instance['label'] = 'NOT_ENOUGH_INFO'

        instance.pop('evidence_label')
        return instance

    random.seed(1)

    names: List[str] = list(map(lambda x: get_scifact_id2label()[x], range(3)))
    dataset_claims_features: Features = Features({
        'id': datasets.Value(dtype='int32', id=None),
        'claim': datasets.Value(dtype='string', id=None),
        'evidence_doc_id': datasets.Value(dtype='string', id=None),
        'label': datasets.ClassLabel(num_classes=3, names=names),
        'evidence_sentences': datasets.features.Sequence(feature=datasets.Value(dtype='int32', id=None), length=-1, id=None),
        'cited_doc_ids': datasets.features.Sequence(feature=datasets.Value(dtype='int32', id=None), length=-1, id=None),
        'title': datasets.Value(dtype='string', id=None),
        'evidence_text': datasets.features.Sequence(feature=datasets.Value(dtype='string', id=None), length=-1, id=None),
        'evidence_full_passage': datasets.Value(dtype='string', id=None)
    })

    # Test has no labels
    data_subsets: Dict[str, Dataset] = dict()

    for split in ['train', 'validation']:
        dataset = scifact_claims[split]
        claim_passage_instances = list(map(to_claim_passage_instance, dataset))

        if split == 'train':
            random.shuffle(claim_passage_instances)
            dev = claim_passage_instances[:200]
            train = claim_passage_instances[200:]

            data_subsets['train'] = records_to_dataset(train, dataset_claims_features)
            data_subsets['validation'] = records_to_dataset(dev, dataset_claims_features)
        else:
            data_subsets['test'] = records_to_dataset(claim_passage_instances, dataset_claims_features)
    return DatasetDict(data_subsets)

