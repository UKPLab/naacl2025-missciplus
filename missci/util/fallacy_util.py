from typing import List, Dict, Optional, Iterable, Tuple


def get_valid_fallacy_names(include_other: bool = True) -> List[str]:
    """
    Only these are applicable for ealuation.
    """
    fallacy_names: List[str] = [
        'Ambiguity',
        'Biased Sample Fallacy',
        'Causal Oversimplification',
        'Fallacy of Division/Composition',
        'Fallacy of Exclusion',
        'False Dilemma / Affirming the Disjunct',
        'False Equivalence',
        'Hasty Generalization',
        'Impossible Expectations'
    ]

    if include_other:
        fallacy_names.append('Other')

    return fallacy_names


