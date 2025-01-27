from collections import Counter
from typing import Optional, Dict, List

from datasets import Dataset

from missci.premise_judge.premise_judge import PremiseJudge


class BaselinePremiseJudge(PremiseJudge):

    def reset_model(self):
        self.predict_always = None

    def __init__(self, setting: str):
        super().__init__()
        self.setting: str = setting
        self.predict_always: Optional[bool] = None

    def fit(self, dataset: Dataset):
        if self.setting == 'match':
            self.predict_always = True
        elif self.setting == 'no-match':
            self.predict_always = False
        else:
            assert self.setting == 'majority'
            self.predict_always = Counter(list(dataset['label'])).most_common()[0][0]

    def predict_instance(self, generated: str, reference: str, instance: Optional[Dict] = None):
        return {
            'predicted': self.predict_always,
            'generated': generated,
            'reference': reference,
            'sample': instance
        }
