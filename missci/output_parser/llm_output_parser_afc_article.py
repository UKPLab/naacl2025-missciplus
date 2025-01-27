import re
from typing import List, Optional


class AFCArticleParser:
    def __init__(self, verdict_labels: List[str], verdict_pattern: str):
        self.verdict_labels: List[str] = list(map(lambda v: v.lower(), verdict_labels))

        if verdict_pattern == 'verdict':
            self.verdict_line_regexp: re.Pattern = re.compile(r'^.*verdict: (.+)$', re.IGNORECASE | re.MULTILINE)
        elif verdict_pattern == 'veracity':
            self.verdict_line_regexp: re.Pattern = re.compile(r'^.*veracity: (.+)$', re.IGNORECASE | re.MULTILINE)
        else:
            raise NotImplementedError(verdict_pattern)

    def parse(self, answer_text: str) -> str:

        verdicts: List[str] = list(map(lambda m: m.group(1).lower(),  self.verdict_line_regexp.finditer(answer_text)))
        verdict: str = self._get_final_verdict(verdicts)
        if verdict is None and len(verdicts) > 0:
            print('Found no match of', verdicts)
            print('Article:', answer_text)
            raise ValueError()
        return verdict

    def _get_final_verdict(self, verdicts: List[str]) -> Optional[str]:

        # Prioritize later verdicts over earlier (if multiple exist)
        for verdict in verdicts[::-1]:

            # Run over sorted to prioritize longer matches (Mostly True) of smaller matches (True)
            # Do this redundantly here for clarity.
            for lbl in sorted(self.verdict_labels, key=lambda l: -len(l)):
                if lbl in verdict:
                    return lbl

        # Default to last verdict
        if len(verdicts) > 0:
            return verdicts[-1].replace('.', '').strip()

        return None
