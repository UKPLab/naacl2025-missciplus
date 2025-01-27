from collections import defaultdict
from os.path import join
from typing import Optional, List, Dict, Iterable, Set

from datasets import Dataset, Features, Value, Sequence, ClassLabel

from missci.util.fileutil import read_jsonl
from missci.util.passage_util import has_mapped_p0, get_p0_from_instance

DEFAULT_DIRECTORY_PATH: str = './dataset'
DATASET_NAME_MAPPED_TEST: str = 'test.missciplus.jsonl'
DATASET_NAME_MAPPED_DEV: str = 'dev.missciplus.jsonl'


def get_mapping_labels() -> List[str]:
    return ['not-entailed', 'entailed']


class MappedDataLoader:
    def __init__(self, dataset_directory: Optional[str] = None):
        self.dataset_directory: str = dataset_directory or DEFAULT_DIRECTORY_PATH
        self.passage_dataset_features: Features = Features({
            'id': Value(dtype='string'),
            'argument_id': Value(dtype='string'),
            'passage_id': Value(dtype='string'),
            'claim': Value(dtype='string'),
            'passage_section': Value(dtype='string'),
            'passage_subsection': Value(dtype='string'),
            'passage_text': Value(dtype='string'),
            'passage_sentences': Sequence(feature=Value(dtype='string')),
            'hidden_premises': Sequence(feature=Value(dtype='string')),
            'label': ClassLabel(num_classes=2, names=get_mapping_labels())
        })

        self.passage_dataset_features_claim_passage: Features = Features({
            'id': Value(dtype='string'),
            'argument_id': Value(dtype='string'),
            'passage_id': Value(dtype='string'),
            'evidence_idx': Value(dtype='int32'),
            'claim': Value(dtype='string'),
            'passage_section': Value(dtype='string'),
            'passage_subsection': Value(dtype='string'),
            'evidence_full_passage': Value(dtype='string'),
            'accurate_premise': Value(dtype='string')
        })

    def load_raw_arguments(self, split: str) -> List[Dict]:
        assert split in {'dev', 'test'}, f'Unknown split: "{split}"'

        file_name: str = DATASET_NAME_MAPPED_TEST if split == 'test' else DATASET_NAME_MAPPED_DEV
        return list(read_jsonl(join(self.dataset_directory, file_name)))

    def load_p0_mappings(self, split: str) -> List[Dict]:
        instances = list(filter(has_mapped_p0, self.load_raw_arguments(split)))
        return instances

    def load_claim_evidence_data(
            self, split: str, level: str, add_p0_to: Optional[str], add_p0_as_passage: bool
    ) -> Dataset:

        if level not in {'sent', 'passage'}:
            raise ValueError(level)

        def create_claim_passage_mapping(instance: Dict) -> List[Dict]:
            arg: Dict = instance['argument']
            arg_id: str = instance['id']
            claim: str = arg['claim']

            if add_p0_as_passage:
                p0: str = get_p0_from_instance(instance, add_p0_as_passage=add_p0_as_passage)
            else:
                p0: str = arg['accurate_premise_p0']['premise']

            if add_p0_to not in {None, 'claim', 'evidence'}:
                raise NotImplementedError(add_p0_to)

            if add_p0_to == 'claim':
                claim = p0 + ' Therefore: ' + claim

            accurate_premise: str = arg['accurate_premise_p0']['premise']

            for passage_key in sorted(list(instance['study']['all_passages'].keys())):
                passage: Dict = instance['study']['all_passages'][passage_key]
                section: str = ''
                subsection: str = ''
                if 'section' in passage and passage['section'] is not None:
                    section = passage['section']
                if 'subsection' in passage and passage['subsection'] is not None:
                    subsection = passage['subsection']

                if level == 'passage':
                    evidence: List[str] = [' '.join(passage['sentences'])]
                else:
                    evidence: List[str] = passage['sentences']

                for i, sentence in enumerate(evidence):

                    if add_p0_to == 'evidence':
                        sentence = p0 + ' ' + sentence

                    yield {
                        'id': f'{arg_id}:{passage_key}',
                        'argument_id': arg_id,
                        'passage_id': passage_key,
                        'evidence_idx': i,
                        'claim': claim,
                        'passage_section': section,
                        'passage_subsection': subsection,
                        'evidence_full_passage': sentence,
                        'accurate_premise': accurate_premise
                    }

        instances: Iterable[Dict] = self.load_raw_arguments(split)
        claim_passage_mappings: List[Dict] = [
            mapping for instance in instances for mapping in create_claim_passage_mapping(instance)
        ]

        # Convert to dataset
        dataset_dict: Dict[str, List] = defaultdict(list)
        for mapped_instance in claim_passage_mappings:
            for key in mapped_instance:
                dataset_dict[key].append(mapped_instance[key])

        return Dataset.from_dict(dataset_dict, self.passage_dataset_features_claim_passage)

    def load_p0_mapping_dataset(self, split: str = 'test', use_full_study: bool = False) -> Dataset:

        def create_claim_passage_mapping(instance: Dict) -> List[Dict]:
            arg: Dict = instance['argument']
            arg_id: str = instance['id']
            claim: str = arg['claim']
            hidden_premises = arg['hidden_premises']

            mapped_passages: Set[str] = set(map(lambda x: x['passage'], arg['accurate_premise_p0']['mapping']))
            use_passage_key: str = 'all_passages' if use_full_study else 'selected_passages'

            for passage_key in sorted(list(instance['study'][use_passage_key].keys())):
                passage: Dict = instance['study'][use_passage_key][passage_key]

                mapping_label: str = 'entailed' if passage_key in mapped_passages else 'not-entailed'

                section: str = ''
                subsection: str = ''
                if 'section' in passage and passage['section'] is not None:
                    section = passage['section']
                if 'subsection' in passage and passage['subsection'] is not None:
                    subsection = passage['subsection']

                yield {
                    'id': f'{arg_id}:{passage_key}',
                    'argument_id': arg_id,
                    'passage_id': passage_key,
                    'claim': claim,
                    'passage_section': section,
                    'passage_subsection': subsection,
                    'passage_text': ' '.join(passage['sentences']),
                    'passage_sentences': passage['sentences'],
                    'hidden_premises': hidden_premises,
                    'label': mapping_label
                }

        instances: Iterable[Dict] = filter(has_mapped_p0, self.load_raw_arguments(split))

        # Create each claim/passage mapping as one instance
        claim_passage_mappings: List[Dict] = [
            mapping for instance in instances for mapping in create_claim_passage_mapping(instance)
        ]

        # Convert to dataset
        dataset_dict: Dict[str, List] = defaultdict(list)
        for mapped_instance in claim_passage_mappings:
            for key in mapped_instance:
                dataset_dict[key].append(mapped_instance[key])

        return Dataset.from_dict(dataset_dict, self.passage_dataset_features)
