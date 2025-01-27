import random
from os.path import join
from typing import List, Optional, Dict

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

from missci.premise_judge.premise_judge import PremiseJudge
from missci.util.fileutil import read_json


def make_premise_text(generated: str, reference: str,
                      accurate_premise: Optional[str] = None, claim: Optional[str] = None, context: Optional[str] = None
                      ) -> str:
    prefix: str = ''
    if context == accurate_premise:
        context = None

    if accurate_premise is not None or context is not None:
        prefix += f'Context: "'
        if accurate_premise is not None:
            prefix += accurate_premise
        if context is not None:
            prefix += f' {context}'
        prefix += '"'

    txt: str = f"""
    {prefix}

    Premises:
1. "@@p1@@"
2. "@@p2@@"
""".strip().replace('@@p1@@', generated).replace('@@p2@@', reference)

    if claim is not None:
        txt += f'\n\nClaim: "{claim}"'

    return txt + '\nQuestion: Do both premises use the same flawed reasoning (fallacy)?'


def make_assistant_reply(label: bool) -> str:
    return 'match' if label else 'no-match'


class Llama3Judge(PremiseJudge):

    def reset_model(self):
        self.icl_examples = []

    def __init__(self,
                 setting: str,
                 llama_type: str,
                 llama_size: str,
                 instructions: str,
                 run_8bit: bool = True,
                 llm_config_path: str = 'llm-config.json',
                 temperature: float = None,
                 samples_per_cls: int = 0,
                 add_context: bool = False,
                 add_claim: bool = False
                 ):
        super().__init__()
        self.setting: str = setting
        assert setting in {'zeroshot', 'icl'}
        self.instructions: str = instructions
        self.icl_examples: List = []
        self.temperature: Optional[float] = temperature
        self.samples_per_cls: int = samples_per_cls
        self.add_context: bool = add_context
        self.add_claim: bool = add_claim

        # Init model
        cache_dir: str = read_json(llm_config_path)[llama_type]['directory']
        key2llama = {'70b': '70B-Instruct', '8b': '8B-Instruct'}
        stored_model_path: str = join(cache_dir, key2llama[llama_size])
        llama = AutoModelForCausalLM.from_pretrained(
            stored_model_path,
            device_map="auto",
            load_in_8bit=run_8bit
        )

        self.pipeline_model = pipeline(
            'text-generation',
            model=llama,
            tokenizer=AutoTokenizer.from_pretrained(stored_model_path),
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def fit(self, dataset: Dataset):
        if self.setting == 'icl':
            assert self.samples_per_cls > 0
            self.icl_examples = []
            dataset = dataset.shuffle()

            num_true: int = 0
            num_false: int = 0
            for sample in dataset:
                if sample['label'] and num_true < self.samples_per_cls:
                    self.icl_examples.append(sample)
                    num_true += 1
                elif not sample['label'] and num_false < self.samples_per_cls:
                    self.icl_examples.append(sample)
                    num_false += 1

            assert len(self.icl_examples) == 2 * self.samples_per_cls
            random.shuffle(self.icl_examples)
            self.icl_examples = [{
                'user': self._make_text(s['generated_premise'], s['reference_premise'], s),
                'assistant': make_assistant_reply(s['label'])
            } for s in self.icl_examples]

    def _make_text(self, generated: str, reference: str, sample: Dict) -> str:
        accurate_premise: Optional[str] = None
        context: Optional[str] = None
        claim: Optional[str] = None

        if self.add_context:
            accurate_premise = sample['p0']
            context = sample['context'] if sample['context'] != accurate_premise else None
        if self.add_claim:
            claim = sample['claim']
        return make_premise_text(generated, reference, accurate_premise, claim, context)

    def predict_instance(self, generated: str, reference: str, instance: Optional[Dict] = None):

        if self.setting == 'zeroshot':
            prompt_data: Dict = self.make_zero_shot(self._make_text(generated, reference, instance))
        elif self.setting == 'icl':
            prompt_data: Dict = self.make_icl_shot(self._make_text(generated, reference, instance))
        else:
            raise NotImplementedError()

        return {
            'predicted': prompt_data['output']['predicted'],
            'generated': generated,
            'reference': reference,
            'sample': instance,
            'prompt_data': prompt_data
        }

    def make_zero_shot(self, sample_text: str) -> Dict:
        messages = [
            {"role": "user", "content": self.instructions + '\n' + sample_text},
        ]
        prompt = self.pipeline_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        output: Dict = self._prompt_llm(prompt)
        return {
            'messages': messages,
            'prompt': prompt,
            'output': output
        }

    def make_icl_shot(self, sample_text: str) -> Dict:
        assert len(self.icl_examples) > 0

        messages = [
            {"role": "user", "content": self.instructions + '\n' + self.icl_examples[0]['user']},
            {"role": "assistant", "content": self.icl_examples[0]['assistant']},
        ]
        for icl_sample in self.icl_examples[1:]:
            messages.append({"role": "user", "content": icl_sample['user']})
            messages.append({"role": "assistant", "content": icl_sample['assistant']})
        messages.append({"role": "user", "content": sample_text})

        prompt = self.pipeline_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        output: Dict = self._prompt_llm(prompt)
        return {
            'messages': messages,
            'prompt': prompt,
            'output': output
        }

    def _prompt_llm(self, prompt: str) -> Dict:
        terminators = [
            self.pipeline_model.tokenizer.eos_token_id,
            self.pipeline_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.pipeline_model(
            prompt,
            eos_token_id=terminators,
            temperature=self.temperature
        )
        complete_output_text: str = outputs[0]["generated_text"]
        answer_text: str = complete_output_text[len(prompt):]

        no_match_keys: List[str] = [
            'no-match', 'no match', 'not match'
        ]
        if len([k for k in no_match_keys if k in answer_text.lower()]) > 0:
            predicted_label: bool = False
        elif 'match' in answer_text.lower():
            predicted_label: bool = True
        else:
            print('Could not parse:')
            print(answer_text)
            print('----\n')
            predicted_label: bool = False

        return {
            'predicted': predicted_label,
            'complete_output_text': complete_output_text,
            'answer': answer_text
        }
