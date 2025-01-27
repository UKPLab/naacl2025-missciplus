import random
from typing import Dict, List, Optional, Set


def get_fallacies_from_argument(instance: Dict) -> List[Dict]:

    fallacies: List[Dict] = []
    for fallacy in instance['argument']['fallacies']:
        for specific_fallacy in fallacy['interchangeable_fallacies']:
            fallacies.append(specific_fallacy)
    return fallacies


def get_p0_from_instance(instance: Dict, add_p0_as_passage: bool, add_section_title: bool = False) -> str:
    arg: Dict = instance['argument']
    p0: str = arg['accurate_premise_p0']['premise']
    if add_p0_as_passage:
        # Locate all passages linked to p0
        mapped_ids: List[str] = list(map(lambda x: x['passage'], arg['accurate_premise_p0']['mapping']))
        if len(mapped_ids) == 0:
            return p0
        else:
            random.shuffle(mapped_ids)
            passage: Dict = instance['study']['selected_passages'][mapped_ids[0]]
            return get_passage_text(passage, add_section_title=add_section_title)
    else:
        return p0


def get_passage_text(passage: Dict, add_section_title: bool = False) -> str:
    passage_text: str = ' '.join(passage['sentences'])
    if add_section_title and len(passage["section"]) > 0:
        # During prompt engingeering this was "(Section: {passage["section"]})\n{passage_text}" on dev
        passage_text = f'(Section: {passage["section"]}) {passage_text}'
    return passage_text


def get_p0_passages(instance: Dict, add_section_title: bool, use_p0_as_backup: bool) -> List[Dict]:
    p0: Dict = instance['argument']['accurate_premise_p0']
    if len(p0['mapping']) == 0:
        if use_p0_as_backup:
            return [{
                'text': p0['premise'],
                'passage': None
            }]
        else:
            return []
    else:
        passages: List[Dict] = []
        for m in p0['mapping']:
            passage: Dict = instance['study']['selected_passages'][m['passage']]
            passage_text = get_passage_text(passage, add_section_title)
            passages.append({
                'text': passage_text,
                'passage': m['passage']
            })
        return passages


def get_sorted_passages(instance: Dict, from_full_study: bool = False) -> List[str]:
    key: str = 'all_passages' if from_full_study else 'selected_passages'
    passages: List[str] = list(instance['study'][key].keys())
    return sorted(passages, key=lambda x: int(x.split('-')[-1]))


def has_mapped_p0(instance: Dict) -> bool:
    return len(instance['argument']['accurate_premise_p0']['mapping']) > 0


def get_gold_fallacy_mapping_dict(instance: Dict, add_empty_context: bool = True) -> Dict[str, bool]:
    """
    This ignores mapping based on the already identified p0.
    :param add_empty_context:
    :param instance:
    :return:
    """
    gold_mapping_fallacies: Dict[str, bool] = {
        passage: False for passage in instance['study']['selected_passages'].keys()
    }

    for fallacy in instance['argument']['fallacies']:
        for mapping in fallacy['mapping']:
            gold_mapping_fallacies[mapping['passage']] = True

    fallacies_from_p0_context: List[Dict] = list(
        filter(lambda x: len(x['fallacy_context']) == 0, instance['argument']['fallacies'])
    )
    if add_empty_context and len(fallacies_from_p0_context) > 0:
        for mapping in instance['argument']['accurate_premise_p0']['mapping']:
            gold_mapping_fallacies[mapping['passage']] = True

    return gold_mapping_fallacies


def get_gold_passages(mappings: List[Dict]) -> List[str]:
    return list(map(lambda x: x['passage'], mappings))


def get_passage_to_fallacies(instance: Dict, use_full_study: bool = True, add_empty_context: bool = True) -> Dict[str, List[str]]:
    if use_full_study:
        passages_key: str = 'all_passages'
    else:
        passages_key: str = 'selected_passages'

    fallacy_dict: Dict[str, List[str]] = {
        key: [] for key in list(instance['study'][passages_key].keys())
    }

    for fallacy in instance['argument']['fallacies']:
        fallacy_id: str = fallacy['id']
        for mapping in fallacy['mapping']:
            fallacy_dict[mapping['passage']].append(fallacy_id)

    fallacies_from_p0_context: List[Dict] = list(
        filter(lambda x: len(x['fallacy_context']) == 0, instance['argument']['fallacies'])
    )
    if add_empty_context:
        for mapping in instance['argument']['accurate_premise_p0']['mapping']:
            for fallacy in fallacies_from_p0_context:
                fallacy_dict[mapping['passage']].append(fallacy['id'])

    return fallacy_dict


def get_passage_mapping_dict_for_fallacies(instance: Dict, use_full_study: bool = True):
    passage_to_fallacies: Dict[str, List] = get_passage_to_fallacies(instance, use_full_study=use_full_study)
    return {
        key: len(passage_to_fallacies[key]) > 0 for key in passage_to_fallacies.keys()
    }


def get_context_mapping_dict_for_fallacies(instance: Dict, assume_accurate_premise_linked: bool = True):
    assert assume_accurate_premise_linked

    return {
        f['id']: len(f['mapping']) > 0 or len(f['fallacy_context'].strip()) == 0  # the second assume p0 is mapped
        for f in instance['argument']['fallacies']
    }


def get_fallacy_to_passages_mapping(instance: Dict, add_interchangeable_fallacies: bool = True) -> Dict[str, List[str]]:
    arg: Dict = instance['argument']
    p0_mapping: List[str] = list(map(lambda x: x['passage'], arg['accurate_premise_p0']['mapping']))

    result: Dict[str, List[str]] = dict()
    for fallacy in arg['fallacies']:
        if len(fallacy['fallacy_context'].strip()) == 0:
            mapping = p0_mapping
        else:
            mapping = list(map(lambda x: x['passage'], fallacy['mapping']))

        assert fallacy['id'] not in result
        result[fallacy['id']] = mapping

        if add_interchangeable_fallacies:
            for i_fallacy in fallacy['interchangeable_fallacies']:
                assert i_fallacy['id'] not in result
                result[i_fallacy['id']] = mapping
    return result

