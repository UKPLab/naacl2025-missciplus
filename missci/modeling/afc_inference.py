import os
import random
from os.path import join, exists
from typing import Dict, Optional, List

import numpy as np
import torch
from datasets import DatasetDict
from torch.nn import Softmax
from transformers import AutoModelForSequenceClassification, Trainer

from missci.data.afc_loader import get_dataset_and_int2lbl
from missci.data.load_covidfact import get_covid_fact_id2label
from missci.data.load_healthver import get_healthver_id2label
from missci.data.load_scifact import get_scifact_id2label
from missci.data.mapped_missci_data_loader import MappedDataLoader
from missci.eval.afc_eval import evaluate_afc_prediction_file
from missci.modeling.afc_training import get_tokenizer_for_task, tokenize_train_dev_test
from missci.util.fileutil import write_jsonl
from missci.util.model_util import get_single_directory_checkpoint


def get_id2label(model_name, is_path: bool = True) -> Dict[int, str]:
    if is_path:
        head, tail = os.path.split(model_name)
    else:
        head = model_name
    if head == 'scifact':
        return get_scifact_id2label()
    elif head == 'covidfact':
        return get_covid_fact_id2label()
    elif head == 'healthver':
        return get_healthver_id2label()
    elif head == 'sci-health-cov':
        return get_scifact_id2label()
    else:
        raise NotImplementedError(model_name)


def afc_inference(
        model_name: str,
        target_task: str,
        prediction_directory: str,
        model_directory: str,
        add_p0_to: Optional[str] = None,
        add_p0_as_passage: bool = False,
        seed: Optional[int] = None
):

    if seed is not None:
        random.seed(seed)

    if target_task == 'missci':
        dataset: DatasetDict = DatasetDict({
            'test': MappedDataLoader().load_claim_evidence_data(
                'test', level='passage', add_p0_to=add_p0_to, add_p0_as_passage=add_p0_as_passage
            ),
            'validation': MappedDataLoader().load_claim_evidence_data(
                'dev', level='passage', add_p0_to=add_p0_to, add_p0_as_passage=add_p0_as_passage
            )
        })
        int2lbl_task = None
    else:
        dataset, int2lbl_task = get_dataset_and_int2lbl(target_task)

    int2lbl_model: Dict[int, str] = get_id2label(model_name)

    tokenizer = get_tokenizer_for_task(target_task)
    tokenized_train, tokenized_dev, tokenized_test = tokenize_train_dev_test(dataset, tokenizer)

    pred_directory: str = join(prediction_directory, model_name)

    model_path: str = get_single_directory_checkpoint(join(model_directory, model_name))
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(
        model=model
    )

    splits = filter(
        lambda x: x[1] is not None,
        [('dev', tokenized_dev), ('test', tokenized_test)]
    )
    for split_name, data in splits:
        softmax = Softmax(dim=1)
        logits, _, _ = trainer.predict(data)
        predicted_probabilities = softmax(torch.FloatTensor(logits)).numpy()
        predicted_labels = map(int, np.argmax(logits, axis=-1))
        predicted_labels = list(map(lambda x: int2lbl_model[x], predicted_labels))

        assert len(predicted_labels) == len(logits)
        assert len(predicted_probabilities) == len(logits)
        assert len(data['id']) == len(logits)

        # Collect predictions
        entries: List[Dict] = []
        for i, entry in enumerate(data):
            entry['prediction'] = {
                'label': predicted_labels[i],
                'probabilities': list(map(float, predicted_probabilities[i])),
                'logits': list(map(float, logits[i]))
            }
            if int2lbl_task is not None:
                entry['label_name'] = int2lbl_task[entry['label']]
            entries.append(entry)

        if not exists(pred_directory):
            os.makedirs(pred_directory)

        raw_prediction_file: str = join(pred_directory, f'{split_name}__{target_task}.jsonl')

        if add_p0_to is not None:
            appendix: str = add_p0_to
            if add_p0_as_passage:
                appendix += '-passage'
            raw_prediction_file = raw_prediction_file.replace('.jsonl', f'.p0-{appendix}.jsonl')

        if seed is not None:
            raw_prediction_file.replace('.jsonl', f'.s{seed}.jsonl')
        write_jsonl(raw_prediction_file, entries)

        evaluate_afc_prediction_file(target_task, raw_prediction_file, split_name)
