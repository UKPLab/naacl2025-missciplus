from os import makedirs
from os.path import join, exists
from typing import Dict, List

import evaluate
import numpy as np
from sklearn.metrics import classification_report
from transformers import set_seed, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from missci.data.afc_loader import get_dataset_and_int2lbl
from missci.util.fileutil import write_jsonl, write_json


class AccuracyMetric:
    def __init__(self, id2lbl: Dict[int, str]):
        self.accuracy = evaluate.load('accuracy')
        self.id2lbl: Dict[int, str] = id2lbl

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = self.accuracy.compute(predictions=predictions, references=labels)
        return acc


def get_output_directories(task_name: str, afc_directory: str):
    output_directory: str = f'{afc_directory}/{task_name}/'
    pred_task_directory: str = f'./predictions/afc/{task_name}/'

    if not exists(output_directory):
        makedirs(output_directory)

    if not exists(pred_task_directory):
        makedirs(pred_task_directory)

    return output_directory, pred_task_directory


def tokenize_train_dev_test(dataset, tokenizer):
    def tokenize_full_passage(instances):
        encodings = tokenizer(
            instances['claim'],
            instances['evidence_full_passage'],
            add_special_tokens=True,
            truncation='only_second',
            padding='max_length',
            return_tensors="pt",
        )
        return encodings

    if 'train' in dataset:
        tokenized_train = dataset['train'].map(tokenize_full_passage, batched=True)
    else:
        tokenized_train = None

    if 'validation' in dataset:
        tokenized_dev = dataset['validation'].map(tokenize_full_passage, batched=True)
    else:
        tokenized_dev = None

    tokenized_test = dataset['test'].map(tokenize_full_passage, batched=True)

    return tokenized_train, tokenized_dev, tokenized_test


def get_tokenizer_for_task(task_name: str, model_name: str = 'microsoft/deberta-v3-large'):
    max_token_dict = {
        'scifact': 1071,
        'healthver': 300,
        'covidfact': 950,
        'sci-health-cov': 1071,
        'missci': 1200
    }

    tokenizer_config: Dict = {
        'pretrained_model_name_or_path': model_name,
        'max_len': max_token_dict[task_name]
    }
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    return tokenizer


def train_and_eval_afc(
    task: str,
    model_name: str,
    seed: int,
    batch_size: int,
    lr: float,
    out_name: str,
    dest_dir: str
):
    assert task in {'scifact', 'healthver', 'covidfact', 'scinli', 'sci-health-cov'}

    set_seed(seed)
    dataset, int2lbl = get_dataset_and_int2lbl(task, seed)
    tokenizer = get_tokenizer_for_task(task)
    num_labels: int = 2 if task == 'covidfact' else 3
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Prepare datasets
    tokenized_train, tokenized_dev, tokenized_test = tokenize_train_dev_test(dataset, tokenizer)

    # Train model
    output_directory, pred_task_directory = get_output_directories(task, dest_dir)
    training_args = TrainingArguments(
        output_dir=join(output_directory, out_name),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        num_train_epochs=5,
        metric_for_best_model='accuracy',
        seed=seed,
        data_seed=seed,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=int(batch_size // 4),
        learning_rate=lr
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        compute_metrics=AccuracyMetric(int2lbl).compute_metrics
    )

    trainer.train()

    for split, ds in [('train', tokenized_train),
                      ('dev', tokenized_dev),
                      ('test', tokenized_test)]:
        logits, labels, metrics = trainer.predict(ds)

        predictions = list(map(int, np.argmax(logits, axis=-1)))
        labels = labels.tolist()
        ids: List[int] = list(map(int, ds['id']))

        if task == 'covidfact':
            target_names = [int2lbl[i] for i in sorted(list(int2lbl.keys()))]
        else:
            target_names = [int2lbl[i] for i in sorted(list(int2lbl.keys()))]
        dataset_metrics: Dict = classification_report(
            labels, predictions, target_names=target_names, output_dict=True, zero_division=0
        )

        print('Results on', split)
        print(classification_report(
            labels, predictions, target_names=target_names, zero_division=0
        ))

        assert len(ids) == len(predictions), f'expected: {len(ids)}, found {len(predictions)}'
        assert len(labels) == len(predictions), f'expected: {len(labels)}, found {len(predictions)}'
        assert len(logits) == len(predictions), f'expected: {len(ids)}, found {len(predictions)}'

        out_name_metric: str = f'{out_name}__{split}.metric.json'
        out_name_preds: str = f'{out_name}__{split}.pred.jsonl'

        write_json(dataset_metrics, join(pred_task_directory, out_name_metric), pretty=True)

        write_out_predictions: List[Dict] = [
            {'id': ids[i], 'predicted': int2lbl[predictions[i]], 'gold': int2lbl[labels[i]],
             'logits': list(map(float, logits[i]))}
            for i in range(len(ids))
        ]
        write_jsonl(join(pred_task_directory, out_name_preds), write_out_predictions)