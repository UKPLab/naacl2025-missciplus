from typing import Dict

from missci.prompt_generators.pairwise.base_pairwise_template_filler import BasePairwiseTemplateFiller


class BasicPRPFiller(BasePairwiseTemplateFiller):

    KEY_PASSAGE_A: str = '@@passage_a@@'
    KEY_PASSAGE_B: str = '@@passage_b@@'
    KEY_QUERY: str = '@@query@@'

    def __init__(
            self, prompt_template_name: str, llm_architecture: str, add_section_title: bool = False,
            convert_prompt_format: bool = True
    ):
        super().__init__(prompt_template_name, llm_architecture, convert_prompt_format)
        self.add_section_title: bool = add_section_title

    def move_a_to_front(self, response: str, default: bool) -> bool:
        index_a: int = response.find('Passage A')
        index_b: int = response.find('Passage B')

        if index_a < 0 and index_b < 0:
            # We didn't find passage A or B in the response
            print('WARNING (no passage found!):', response)
            return default
        elif index_a < 0:
            # We didn't find passage A in the response
            return False
        elif index_b < 0:
            # We didn't find passage B in the response
            return True
        elif index_a < index_b:
            # Passage A was found in the response, before passage B
            return True
        elif index_b < index_a:
            # Passage A was found in the response, before passage A
            return False
        else:
            raise ValueError('Should not happen!')

    def _fill_template(self, instance: Dict, passage_a: Dict, passage_b: Dict) -> str:

        passage_a_text: str = ' '.join(passage_a['sentences'])
        passage_b_text: str = ' '.join(passage_b['sentences'])

        if self.add_section_title:
            if len(passage_a["section"]) > 0:
                passage_a_text = f'(Section: {passage_a["section"]}) {passage_a_text}'
            if len(passage_b["section"]) > 0:
                passage_b_text = f'(Section: {passage_b["section"]}) {passage_b_text}'

        return self.prompt_template.replace(
            BasicPRPFiller.KEY_QUERY, instance['argument']['claim']
        ).replace(
            BasicPRPFiller.KEY_PASSAGE_A, passage_a_text
        ).replace(
            BasicPRPFiller.KEY_PASSAGE_B, passage_b_text
        )

    def get_name(self, suffix: str):
        name: str = super().get_name(suffix)
        if self.add_section_title:
            name = name.replace('.jsonl', f'.add-section-title.jsonl')
        return name
