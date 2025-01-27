import os
from os.path import join, exists, isdir
from typing import Optional, Dict

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from missci.premise_judge.premise_judge import PremiseJudge
from missci.premise_judge.sft_prompt_formatter import SFTPromptFormatter


class SFTLLMJudge(PremiseJudge):

    @classmethod
    def load(
            cls, model_name: str, instructions: str, run_8bit: bool, add_context: bool = False, add_claim: bool = False
    ):
        return cls(
            output_name=model_name, instructions=instructions, run_8bit=run_8bit,
            add_context=add_context, add_claim=add_claim, load_from_disk=True,
            llm='', size='', num_epochs=-1, batch_size_accum=-1, batch_size_per_gpu=-1,
            scheduler='', lr=-1., lora_alpha=-1, lora_dropout=-1., lora_r=-1
        )

    def __init__(self, llm: str, size: str, instructions: str, num_epochs: int, batch_size_accum: int,
                 batch_size_per_gpu: int, scheduler: str, lr: float, lora_alpha: int, lora_dropout: float, lora_r: int,
                 output_name: str, run_4bit: bool = False,
                 run_8bit: bool = True, llm_config_path: str = 'llm-config.json', temperature: float = None,
                 add_context: bool = False, add_claim: bool = False, model_directory: str = './models/judge/llms',
                 load_from_disk: bool = False
                 ):
        super().__init__()
        self._llm_name: str = llm
        self._size: str = size
        self._instructions: str = instructions
        self._num_epochs: int = num_epochs
        self._batch_size_accum: int = batch_size_accum
        self._batch_size_per_gpu: int = batch_size_per_gpu
        self._scheduler: str = scheduler
        self._lr: float = lr
        self._lora_alpha: int = lora_alpha
        self._lora_dropout: float = lora_dropout
        self._lora_r: int = lora_r
        self._run_8bit: bool = run_8bit
        self._run_4bit: bool = run_4bit
        self._llm_config_path: str = llm_config_path
        self._temperature: float = temperature
        self._add_context: bool = add_context
        self._add_claim: bool = add_claim
        self._output_name = join(model_directory, output_name)

        assert not (run_4bit and run_8bit)

        if load_from_disk:
            assert exists(self._output_name)
            assert isdir(self._output_name)
            self._load_from_disk()
        else:
            if not exists(model_directory):
                os.makedirs(model_directory)

            assert self._run_8bit or self._run_4bit
            if self._run_8bit:
                self._bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True  #, bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                assert self._run_4bit
                self._bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )

            self._llm = None
            self._tokenizer = None
            self._is_trained: bool = False
            self.reset_model()

        assert self._llm is not None
        assert self._tokenizer is not None
        self._formatter: SFTPromptFormatter = SFTPromptFormatter(
            instructions=self._instructions,
            tokenizer=self._tokenizer,
            add_claim=self._add_claim,
            add_context=self._add_context,
            add_p0=self._add_context,
        )

    def predict_instance(self, generated: str, reference: str, instance: Optional[Dict] = None) -> Dict:
        assert self._is_trained
        if instance is None:
            instance = {
                self.key_generated: generated,
                self.key_reference: reference
            }
        else:
            assert reference == instance[self.key_reference]
            assert generated == instance[self.key_generated]

        prompt = self._formatter.format_instance(instance, add_answer=False)
        input_ids = self._tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs_answer = self._llm.generate(input_ids, max_new_tokens=100)[:, input_ids.size()[-1] - 1:-1]
        outputs_answer = self._tokenizer.batch_decode(outputs_answer.detach().cpu().numpy())[0]
        if 'no-match' in outputs_answer or 'no match' in outputs_answer:
            predicted_label = False
        elif 'match' in outputs_answer:
            predicted_label = True
        else:
            predicted_label = False

        return {
            'predicted': predicted_label,
            'generated': instance[self.key_generated],
            'reference': instance[self.key_reference],
            'sample': instance,
            'prompt_data': {
                'prompt': prompt,
                'output': outputs_answer
            }
        }

    def fit(self, dataset: Dataset):
        assert not self._is_trained
        peft_config = LoraConfig(
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            r=self._lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        current_model = prepare_model_for_kbit_training(self._llm)
        current_model = get_peft_model(current_model, peft_config)

        args = TrainingArguments(
            output_dir=self._output_name,
            num_train_epochs=self._num_epochs,
            per_device_train_batch_size=self._batch_size_per_gpu,
            gradient_accumulation_steps=self._batch_size_accum,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=self._lr,
            bf16=True,
            tf32=True,
#            max_grad_norm=0.3,
#            warmup_ratio=0.03,
            lr_scheduler_type=self._scheduler
        )

        trainer = SFTTrainer(
            model=current_model,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=2048,
            tokenizer=self._tokenizer,
            packing=False,
            formatting_func=self._formatter.format,
            data_collator=DataCollatorForCompletionOnlyLM(
                self.get_response_template(), tokenizer=self._tokenizer
            ),
            args=args,
        )
        trainer.train()
        trainer.save_model()

        print('DONE Training. Load LLM')
        self._load_from_disk()
        print('loaded trained LLM.')

    def _load_from_disk(self):
        self._llm = AutoPeftModelForCausalLM.from_pretrained(
            self._output_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_8bit=self._run_8bit,
            load_in_4bit=self._run_4bit
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._output_name)
        self._is_trained = True

    def reset_model(self):
        self._llm = self._create_model()
        self._tokenizer = self._create_tokenizer()
        self._llm.config.pretraining_tp = 1
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        self._is_trained = False

    def _create_model(self) -> AutoModelForCausalLM:
        raise NotImplementedError()

    def _create_tokenizer(self) -> AutoTokenizer:
        raise NotImplementedError()

    def get_response_template(self):
        raise NotImplementedError()