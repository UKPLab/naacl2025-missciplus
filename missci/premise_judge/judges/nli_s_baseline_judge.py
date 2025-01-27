from collections import Counter
from typing import Optional, Dict, List

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score

from missci.premise_judge.premise_judge import PremiseJudge

from sklearn.linear_model import LogisticRegression


class NLIBaselinePremiseJudge(PremiseJudge):

    def reset_model(self):
        self.model = LogisticRegression()

    def __init__(self):
        super().__init__()
        self.model = None
        self.reset_model()

    def fit(self, dataset: Dataset):
        X = np.array(dataset['nli_s']).reshape((-1, 1))
        y = dataset['label']
        self.model.fit(X, y)

    def predict_instance(self, generated: str, reference: str, instance: Optional[Dict] = None):
        X = np.array([instance['nli_s']]).reshape((-1, 1))
        pred = self.model.predict(X)
        return {
            'predicted': bool(pred[0]),
            'generated': generated,
            'reference': reference,
            'sample': instance
        }
