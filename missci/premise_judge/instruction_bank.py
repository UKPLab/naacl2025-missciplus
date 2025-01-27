from typing import Dict, List

INSTRUCTION_BANK: Dict[str, str] = {
    'same-reasoning-1': """
You are given two premises that exhibit some reasoning of a larger argument. Both premises apply a fallacy. Your task is to select whether the reasoning on an abstract level is identical. If one premise is more specific, they can still apply the same false reasoning.
Provide your answer in the first line of your response. Answer with "match" if both premises apply the same false reasoning. Answer with "no-match" if they apply different false reasoning.
    """,

    'same-reasoning-2': """
    I'll present you with two premises, each containing a fallacy in their reasoning. Analyze both statements:
Your task: Determine if the core flawed logic behind the fallacies in both statements is identical.
Respond with "match" if the underlying reasoning is the same, even if the specifics differ. Respond with "no-match" if they represent different fallacies.
""",

    'same-reasoning-3': """
    Task: Analyze both premises and determine if they commit the same type of fallacy.
    Answer: (match / no-match)
""",
    'same-reasoning-4': """
    Determine whether two premises exhibit identical false reasoning, regardless of their specificity.
    Provide your answer in the first line of your response. Answer with "match" if both premises apply the same false reasoning. Answer with "no-match" if they apply different false reasoning.
    """,
 'same-reasoning-5': """
    Task: Analyze both premises and determine if they apply a similar reasoning regardless of specificity.
    Answer: (match / no-match)
"""
}

ADDONS: Dict = {
    'only-context-1': 'Both premises use the content of the "context" field.',
    'only-claim-1': 'Both premises should be used to support the provided claim.',
    'connect-1': """You are also provided with the content of a scientific publication through the "context" field and with a claim drawn as a conclusion from the context.
    Each premise highlights a flawed reasoning pattern (fallacy) in its attempt to connect the scientific publication to the claim."""
}


def get_judge_instruction(prompt_name: str, add_instructs: List[str] = None) -> str:
    instructions: str = INSTRUCTION_BANK[prompt_name].strip()

    if add_instructs is not None and len(add_instructs) > 0:
        notes = '\nNote:'
        for add in add_instructs:
            notes += f'\n{ADDONS[add]}'
        instructions += notes

    return instructions
