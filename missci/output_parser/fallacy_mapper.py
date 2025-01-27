from typing import Dict, List, Set

FALLACY_DICT: Dict[str, List[str]] = {
    'Ambiguity': [
        'ambiguity', 'ambiguous understanding', 'equivocation', 'is ambiguous', 'ambiguous',
        'term x is used to mean y in the premise. term x is used to mean z in the conclusion'
    ],
    'Biased Sample Fallacy': [
        'biased sample', 'selection bias', 'sample fallacy', 'fallacy of sampling', 'sampling fallacy',
        'based on a sample that is biased', 'sampling bias', 'sample size is biased', 'sampling error',
        'only considers a specific population'
    ],
    'Causal Oversimplification': [
        'causal oversimplification', 'false cause', 'false causality', 'fallacy of oversimplification',
        'correlation/causation', 'correlation does not imply causation', 'post hoc ergo propter hoc',
        'correlation-causality', 'false temporal association', 'correlation/causality', 'fallacy of correlation',
        'confounding variables', 'correlation and causality', 'confounding bias', 'fallacy of causation',
        'causal', 'causation', 'post hoc ergo propter', 'correlation implies', 'claim assumes that the vaccine is causing',
        'may be influenced by other factors', 'single cause', 'ignoring the possibility of other factors', 'post hoc fallacy',
        'oversimplified cause'
    ],
    'Fallacy of Division/Composition': ['fallacy of division', 'fallacy of composition', 'composition',
                                        'inferring that something is true of the whole from the fact that it is true of some part of the whole',
                                        'inferring that the whole '
                                        ],
    'Fallacy of Exclusion': [
        'fallacy of exclusion', 'false exclusion', 'exclusion', 'fallacy of omission', 'suppressed evidence',
        'when only select evidence is presented in order', 'excludes relevant evidence',
        'ignoring the relevance of a premise', 'ignorance', 'ignoring the relevance',
        'ignoring the context', 'contextual fallacy', 'contextomy', 'withholding relevant and significant evidence',
        'claim excludes relevant information from premise', 'selectively presents evidence', 'selective presentation',
        'cherry pickin'
    ],
    'False Dilemma / Affirming the Disjunct': [
        'false dilemma', 'affirming the disjunct', 'false negation', 'false negative', 'false exclusivity',
        'affirming the antecedent', 'affirming the consequent', 'false alternatives', 'false dichotomy',
        'false dile mma', 'argument presents only two alternatives', 'black-or-white'
    ],
    'False Equivalence': [
        'false equivalence', 'false analogy', 'false ex equivalence', 'fallacy of analogy', 'false comparative',
        'assumes that two subjects that share a single trait (receiving a vaccine) are equivalent',
        'two subjects that share a single trait are equivalent', 'equivalence.',
        'claim assumes that the vaccine efficacy of 91.3% is equivalent', 'equivalence', 'faulty analogy'
    ],
    'Hasty Generalization': [
        'hasty generalization', 'sweeping generalization', 'false generalization', 'sample size fallacy',
        'insufficient sample size', 'faulty generalization', 'false generalization', 'fallacy of generalization',
        'generalization', 'is too small to draw conclusions', 'based on a limited sample size'
    ],
    'Impossible Expectations': [
        'impossible expectations', 'false expectations', 'false expectation', 'misplace priorities',
        'comparing a realistic solution with an idealized one', 'impossible expectation', 'unrealistic standards',
        'comparing a realistic solution', 'impossibility fallacy'
    ],
    'None': ['none ', 'none.', 'none', 'no fallacy', 'reasonable conclusion', 'directly supports the claim', 'therefore, i output']
}

IGNORE_FALLACIES: Set[str] = {
    'false precision', 'false alarm', 'red herring', 'ad hominem', 'fallacy of misinterpretation',
    'false negativity', 'false premise', 'false universal', 'false correlation', 'false certainty',
    'false reassurance', 'definition', 'definition2', 'hidden assumption', 'extrapolation', 'multiple fallacies',
    'irrelevant information', 'misapplication of evidence', 'false attribution', 'misinterpretation of scientific information',
    'lack of evidence', 'exaggeration', 'false claim', 'lack of context', 'overstatement', 'ignored claim',
    'fallacy of surprise', 'irrelevant thesis', 'speculative reasoning', 'non-sequitur', 'incomplete analysis',
    'fallacy of incomplete evidence', 'fallacy of repetition', 'misleading vividness', 'false necessity',
    'false assurance', 'measurement bias', 'statistical bias', 'negative proof fallacy', 'fallacy of ignorance',
    'unbalanced consideration', 'hasty of composition', 'hasty ginger', 'false positive', 'necessity fallacy',
    'sufficient-necessary', 'fallacy of oversight', 'assumes facts not in evidence', 'hasty logical fallacy',
    'false conclusion', 'fallacy of exaggeration', 'non sequitur', 'wishful thinking', 'unproven assumption',
    'confirmation bias', 'appeal to', 'false positive fallacy', 'false consensus', ' ad hominem', 'unstated assumption',
    'slippery slope', 'false authority', 'false explanation', 'false ex equivalence', 'fallacy class',
    'fallacy of misleading statistics', 'emphasis on uncertainty', 'confirmation bias', 'fallacy of confusion',
    'extrapolation fallacy', 'argument from ignorance', 'fallacy of conclusion', 'speculation', 'fallacy of accident',
    'survivorship bias', 'overgeneralization', 'false hope fallacy', 'false assumption', 'misuse of statistics',
    'fallacy of inference', 'fallacy of com', 'fallacy of authority', 'fallacy of magnitude', 'false hope',
    'fallacy of relevance', 'fear of the unknown', 'bandwagon', 'hypothetical fallacy', 'unproven assumption',
    'false exaggeration', 'accident', 'circular reasoning', 'begging the question', 'loaded question',
    'inconsistent comparison', 'insufficient evidence', 'misleading statistic', 'fallacy of attribution',
    'irrelevant evidence', 'fallacy of misrepresentation', 'oversimplification', 'fear-mongering',
    'overwhelming exception', 'false comparison', 'missing the point', 'fallacy of extrapolation', 'overemphasis',
    'fallacy of coincidence', 'fallacy of assumption', 'false promise', 'fallacy of reversal',
    'fallacy of amplification', 'false experiment fallacy', 'value-judgment fallacy', 'fallacy of emphasis',
    'irrelevant conclusion', 'overconfidence', 'revised premise avoids fallacies', 'fear mongering',
    'fallacy of irrelevance', 'fallacy of definition', 'false experiment', 'incomplete comparison',
    'glittering generalities', 'fallacy of novelty', 'necessary condition', 'sufficient condition', 'definition 1',
    'argument from silence', 'definition', 'overstatement.', 'median time from diagnosis', 'neutralizing antibodies ',
    'fallacious premise', 'assuming that vaccination is the only strategy', 'claim assumes that the vaccine is able to save every life',
    'contextomizer', 'hasty categorical syl', 'value judgment fallacy', 'conflict of interest', 'jumping to conclusion',
    'straw man'
}


class FallacyMapper:
    def __init__(self, fallacy_dict: Dict[str, List[str]] = FALLACY_DICT,
                 ignore_fallacies: Set[str] = IGNORE_FALLACIES):
        self.fallacies: Dict[str, str] = {
            val: key for key in fallacy_dict.keys() for val in fallacy_dict[key]
        }
        self.candidates: List[str] = sorted(list(self.fallacies.keys()), key=lambda f: -len(f))
        self.ignore_fallacies: Set[str] = ignore_fallacies

    def map_fallacy(self, predicted_fallacy_class: str):
        predicted_fallacy_class = predicted_fallacy_class.lower()
        for fallacy_candidate in self.candidates:
            if fallacy_candidate in predicted_fallacy_class:
                return self.fallacies[fallacy_candidate]

        for fallacy in self.ignore_fallacies:
            if fallacy in predicted_fallacy_class:
                return 'Other'
