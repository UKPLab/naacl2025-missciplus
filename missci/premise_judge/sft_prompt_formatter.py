from missci.premise_judge.judges.llama3_judge import make_premise_text


class SFTPromptFormatter:
    def __init__(self, instructions, tokenizer, add_p0=False, add_claim=False, add_context=False):
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.add_p0 = add_p0
        self.add_claim = add_claim
        self.add_context = add_context

    def format(self, sample):
        instance_texts = []

        for i in range(len(sample['generated_premise'])):
            instance_texts.append(self.format_instance({
                k: sample[k][i] for k in sample
            }))
        return instance_texts

    def format_instance(self, sample, add_answer=True):
        p0 = sample['p0'] if self.add_p0 else None
        claim = sample['claim'] if self.add_claim else None
        context = sample['context'] if (self.add_context and sample['context'] != sample['p0']) else None

        prompt = self.instructions + '\n' + make_premise_text(
            sample['generated_premise'], sample['reference_premise'], p0, claim, context
        )

        if add_answer:
            label = 'match' if sample['label'] else 'no-match'
            return self.tokenizer.apply_chat_template([
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': label}
            ],
                tokenize=False, add_generation_prompt=False
            )
        else:
            return self.tokenizer.apply_chat_template([
                {'role': 'user', 'content': prompt}
            ],
                tokenize=False, add_generation_prompt=True
            )
