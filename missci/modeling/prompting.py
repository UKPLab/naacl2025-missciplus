def filled_template_to_prompt_llama(filled_template: str) -> str:
    b_inst: str = " [INST] "
    e_inst: str = " [/INST]"
    s_prompt: str = "<s>"

    if '@@system_prompt@@' in filled_template:
        filled_template = filled_template.replace('@@system_prompt@@', '')

    filled_template = filled_template.strip()

    # Make sure it is completely filled
    if '@@' in filled_template:
        raise ValueError(f'The template still contains unfilled fields: {filled_template}!')

    return s_prompt + b_inst + filled_template + e_inst


def filled_template_to_prompt_llama3(filled_template: str) -> str:
    if '@@' in filled_template:
        raise ValueError(f'The template still contains unfilled fields: {filled_template}!')

    return f'''
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    {filled_template}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    '''


def filled_template_to_prompt_gpt(filled_template: str) -> str:

    if '@@system_prompt@@' in filled_template:
        filled_template = filled_template.replace('@@system_prompt@@', '')

    filled_template = filled_template.strip()

    # Make sure it is completely filled
    if '@@' in filled_template:
        raise ValueError(f'The template still contains unfilled fields: {filled_template}!')

    return filled_template


def to_prompt_for_llm(llm_architecture: str, filled_template: str):
    if llm_architecture == 'llama2':
        prompt: str = filled_template_to_prompt_llama(filled_template)
    elif llm_architecture == 'gpt4' or llm_architecture == 'chatgpt':
        prompt = filled_template_to_prompt_gpt(filled_template)
    else:
        raise ValueError(f'Unknown: "{llm_architecture}"!')
    return prompt
