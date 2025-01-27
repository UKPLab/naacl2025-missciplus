import json
import re
from typing import Dict, List, Tuple

from missci.output_parser.fallacy_mapper import FallacyMapper


def normalize_fallacious_premise(premise: str) -> str:

    if len(premise) > 0 and premise[0] == '"':
        premise = premise[1:]
    if len(premise) > 0 and premise[-1] == '"':
        premise = premise[:-1]
    return premise.strip()


def add_p0_id_if_not_exists(passages: Dict, prediction: Dict, accurate_p0: str) -> Dict:
    if 'p0_id' in prediction:
        return prediction
    else:
        text = prediction['p0'].split('\n')[-1]
        for key in passages:
            passage_text: str = ' '.join(passages[key]['sentences'])
            if passage_text == text:
                prediction['p0_id'] = key
                return prediction
        if prediction['p0'] == accurate_p0:
            prediction['p0_id'] = None
            return prediction

        print(prediction['p0'])
        assert False


class ArgumentReconstructionParser:
    def __init__(self, file_name: str, fallacy_mapper: FallacyMapper):
        self.file_name: str = file_name
        self.fallacy_mapper: FallacyMapper = fallacy_mapper

        self.regular_expressions1: List[Tuple[re.Pattern, int, int]] = [
            (re.compile(
                r'^.*fallacious premise( 3\.?\d?)?: (.+)[;\."] applied fallacy class: (.+)$',
                re.IGNORECASE | re.MULTILINE
            ), 2, 3),
            # ADDED POST DEV
            (re.compile(
                r'^.*fallacious premise( \.?\d?)?: (.+)[;\."]?\s+applied fallacy class(es)?\s*\d?: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 2, 4),
            # END ADDED
            (re.compile(
                r'^.*fallacious premise( 3\.?\d?)?: (.+)[;\."] applied fallacy: (.+)$',
                re.IGNORECASE | re.MULTILINE
            ), 2, 3),
            (re.compile(
                r'^fallacious premise( 3\.?\d?)?: (.+)\n+applied fallacy class: (.+)$',
                re.IGNORECASE | re.MULTILINE
            ), 2, 3),
            (re.compile(
                r'^.*fallacious premise(.?\d?)?: (.+)\n*applied fallacy( class)?: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 2, 4),
            (re.compile(
                r'\*\*fallacious premise:?\*\* (.+)\s+\*\*(applied)? fallacy class:\*\* (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'\d+\. \*\*fallacious premise:?\*\*:? (.+); \*\*(applied) fallacy class:?\*\*:? (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'\*\*fallacious premise \d:?\*\* (.+)\s+\*\*(applied)? fallacy class:\*\* (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'premise \d+:\s+"(.+)"\s+(applied)? fallacy class: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'fallacious premise:\s+(.+)\s+(applied )?fallacy class:\s+(.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3)
        ]

        # Backup that dont truly follow the template.
        self.regular_expressions2: List[Tuple[re.Pattern, int, int]] = [
            (re.compile(
                r'\*\s*"(.+)"; (applied) fallacy class: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'\d+\. "(.+)".*[\s\*]+?(applied)? fallacy class: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'\d+\..+?: (.+)\n(applied) fallacy class: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'\d+\. "(.+)".*?; (applied)? fallacy class: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'^\d+\. \*\*(.+)\*\*: (.+)$',
                re.IGNORECASE | re.MULTILINE
            ), 2, 1),
            (re.compile(
                r'fallacious premise is:\s*(.+)\s*.+?(applied)?fallacy class is (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'\d+\. (.+?): "(.+?)".',
                re.IGNORECASE | re.MULTILINE
            ), 2, 1),
            (re.compile(
                r'\d+\. (.+?): (.+)$\s+applied fallacy class: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 2, 1),
            (re.compile(
                r'\*\*hidden assumption:\*\*\s+(.+)\s+\*\*applied fallacy class:\*\*\s+(.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r'the fallacious premise is: "(.+)"\s+this premise(.+)\s+the applied fallacy class is (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 3),
            (re.compile(
                r'\*\*fallacious premise \d+:\*\*\s+"(.+)"\s+\*\*applied fallacy class:\*\*\s(.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r'the hidden assumption is: "(.+)"\s+.+\s+the applied fallacy class is:\s(.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r'\*\*fallacious premise:?\*\*:?\s(.+)\s+\*\*applied fallacy class:?\*\*:?\s(.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r'premise \d+: "?(.+)"?\s+.*\s*applied fallacy class: (.+)',
                re.IGNORECASE | re.MULTILINE
            ), 1, 2)
        ]

        self.regular_expressions3: List[Tuple[re.Pattern, int, int]] = [
            (re.compile(
                r'^\d+\.\s*"(.+?)".*?\((.+)\)\.$',
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r"fallacious premise( 3\.?\d?)?: (.+)[\d\D]+?(applied)? ?fallacy class: (.+)$",
                re.IGNORECASE | re.MULTILINE
            ), 2, 4),
            (re.compile(
                r"fallacious premise (is)?:[\s\n]*([^\n]+)[\d\D]*?[\s\n]*(the )?(applied fallacy class)( is)?:[\s\n]*(.+)",
                re.IGNORECASE | re.MULTILINE
            ), 2, 6),
            (re.compile(
                r"\d+\. (.+?): (.+)$",
                re.IGNORECASE | re.MULTILINE
            ), 2, 1),
            (re.compile(
                r"fallacious premise: (.+) This is an example of the (.+)",
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r"hidden assumption is:\s+(.+)\s+.+?\s+applied fallacy class: (.+)",
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r"fallacy class is: (.+)\s+.*fallacious premise is: (.+)",
                re.IGNORECASE | re.MULTILINE
            ), 2, 1),
            (re.compile(
                r"the applied fallacy class is (.+)\s+the fallacious premise is: (.+)",
                re.IGNORECASE | re.MULTILINE
            ), 2, 1),
            (re.compile(
                r"hidden assumption:\s+\* (.+)\s+.*\s*applied fallacy class: (.+)",
                re.IGNORECASE | re.MULTILINE
            ), 1, 2),
            (re.compile(
                r"hidden assumption .* is:\s+\* (.+)\s*.*\s*.*?applied fallacy class is: (.+)",
                re.IGNORECASE | re.MULTILINE
            ), 1, 2)
        ]

        self.no_fallacy_strings: List[str] = [
            '"no fallacy"', '"no fallacy."', 'no fallacy'
        ]

        self.regexp_fallacy_list_reply: re.Pattern = re.compile(
            r'fallacious premise: (.+)\n*.*\n*(\d\. (.+):\n?.+)', re.MULTILINE | re.IGNORECASE
        )

    def is_no_fallacy(self, generated_text: str) -> bool:
        for nf_str in self.no_fallacy_strings:
            if nf_str in generated_text.lower():
                return True
        return False

    def parse(self, prediction: Dict) -> Dict:

        # Map to p0 passage ID
        if 'output' not in prediction:
            print(json.dumps(prediction, indent=2))
        if 'output' in prediction['output']:
            output_text: str = prediction['output']['output'] # if 'output' in prediction['output'] else prediction['output']['answer']
        else:
            output_text = prediction['output']['answer']
        predicted_fallacies: List[Dict] = self._get_predicted_fallacies(output_text)

        prediction['is_parsed'] = len(predicted_fallacies) > 0
        prediction['predicted_fallacies'] = predicted_fallacies
        prediction['valid_logic'] = self.is_no_fallacy(output_text)

        num_no_fallacy = len([
            p for p in prediction['predicted_fallacies'] if p['generated_fallacy_class'] is None
        ])
        if num_no_fallacy > 0:
            assert len(prediction['predicted_fallacies']) == 1
            prediction['predicted_fallacies'] = []
            prediction['valid_logic'] = True

        if 'argument_id' in prediction and len(predicted_fallacies) == 0 and not prediction['valid_logic'] and 'fallacy class' in output_text.lower():
            print(self.file_name)
            print('NO FALLACIES', prediction['argument_id'])
            print(output_text)
            print('---\n\n')

        return prediction


    def _parse_fallacy_list_reply(self, generated_text: str) -> List[Dict]:
        match = re.match(self.regexp_fallacy_list_reply, generated_text)
        fallacious_premise: str = match.group(1)
        fallacy_matches = list(re.finditer(r'(\d\. (.+):\n?.+)', generated_text))

        out: List[Dict] = []
        for m in fallacy_matches:
            out.append({
                'sentence': m.group(0),
                'fallacious_premise': fallacious_premise,
                'generated_fallacy_class': m.group(2),
                'parsed': True
            })
        return out

    def _get_predicted_fallacies(self, generated_text: str) -> List[Dict]:
        best_parsed: List[Dict] = []

        # Fix typos
        generated_text = generated_text.replace(
            'Fallicious Premise', 'Fallacious Premise'
        ).replace(
            'Fallacious Premiere:', 'Fallacious Premise:'
        )

        # Try patterns

        for pattern, grp_premise, grp_cls in self.regular_expressions1 + self.regular_expressions2:
            matches = list(re.finditer(pattern, generated_text))
            if len(matches) > len(best_parsed):
                best_parsed = []
                for m in matches:
                    best_parsed.append({
                        'sentence': m.group(0),
                        'fallacious_premise': m.group(grp_premise),
                        'generated_fallacy_class': m.group(grp_cls),
                        'parsed': True,
                        'parsed_via': repr(pattern.pattern)
                    })

        # # Still nothing?
        if len(best_parsed) == 0:
            for pattern, grp_premise, grp_cls in self.regular_expressions3:
                matches = list(re.finditer(pattern, generated_text))
                if len(matches) > len(best_parsed):
                    best_parsed = []
                    for m in matches:
                        best_parsed.append({
                            'sentence': m.group(0),
                            'fallacious_premise': m.group(grp_premise),
                            'generated_fallacy_class': m.group(grp_cls),
                            'parsed': True,
                            'parsed_via': repr(pattern.pattern)
                        })

        # Still nothing
        if len(best_parsed) == 0 and re.match(self.regexp_fallacy_list_reply, generated_text):
            best_parsed = self._parse_fallacy_list_reply(generated_text)

        for parsed in best_parsed:
            parsed['normalized_fallacy_class'] = self.fallacy_mapper.map_fallacy(parsed['generated_fallacy_class'])
            parsed['fallacious_premise'] = normalize_fallacious_premise(parsed['fallacious_premise'])

        for i, parsed in enumerate(best_parsed):
            parsed['per-prompt-position-rank'] = i+1

        return best_parsed
