from typing import List, Dict, Optional

from datasets import Dataset
from tqdm import tqdm


class PremiseJudge:

    def __init__(self, key_generated: str = 'generated_premise', key_reference: str = 'reference_premise'):
        self.key_generated = key_generated
        self.key_reference = key_reference

    def predict_dataset(self, dataset: Dataset) -> List[Dict]:
        predictions: List[Dict] = []
        for sample in tqdm(dataset):
            predictions.append(self.predict_instance(
                sample[self.key_generated], sample[self.key_reference], instance=sample)
            )
        return predictions

    def predict_instance(self, generated: str, reference: str, instance: Optional[Dict] = None) -> Dict:
        raise NotImplementedError()

    def fit(self, dataset: Dataset):
        raise NotImplementedError()

    def reset_model(self):
        raise NotImplementedError()
