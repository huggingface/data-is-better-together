import json
from typing import Any, Dict, List

from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.tasks.text_generation import TextGeneration
from distilabel.steps import StepInput, StepOutput, Step

from dotenv import load_dotenv

from defaults import (
    DEFAULT_DOMAIN,
    DEFAULT_PERSPECTIVES,
    DEFAULT_TOPICS,
    DEFAULT_EXAMPLES,
    DEFAULT_SYSTEM_PROMPT,
    N_PERSPECTIVES,
    N_TOPICS,
    N_EXAMPLES,
)

load_dotenv()

# Application description used for SelfInstruct
APPLICATION_DESCRIPTION = f"""You are an AI assistant than generates queries around the domain of {DEFAULT_DOMAIN}.
Your should not expect basic but profound questions from your users.
The queries should reflect a diversity of vision and economic positions and political positions.
The queries may know about different methods of {DEFAULT_DOMAIN}.
The queries can be positioned politically, economically, socially, or practically.
Also take into account the impact of diverse causes on diverse domains."""


TOPICS = DEFAULT_TOPICS[:N_TOPICS]
PERSPECTIVES = DEFAULT_PERSPECTIVES[:N_PERSPECTIVES]
EXAMPLES = DEFAULT_EXAMPLES[:N_EXAMPLES]

QUESTION_EXAMPLES_PROMPT = """ Examples of high quality questions:"""
ANSWER_EXAMPLES_PROMPT = """ Examples of high quality answers:"""

for example in EXAMPLES:
    QUESTION_EXAMPLES_PROMPT += f"""\n- Question: {example["question"]}\n"""
    ANSWER_EXAMPLES_PROMPT += f"""\n- Answer: {example["answer"]}\n"""


def create_topics(topics: List[str], positions: List[str]) -> List[str]:
    return [
        f"{topic} from a {position} perspective"
        for topic in topics
        for position in positions
    ]


class DomainExpert(TextGeneration):
    """A customized task to generate text as a domain expert in the domain of farming and agriculture."""

    _system_prompt: (str) = DEFAULT_SYSTEM_PROMPT
    _template: str = (
        """{instruction}\nThis is the the instruction.\n Examples: """
        + QUESTION_EXAMPLES_PROMPT
        + ANSWER_EXAMPLES_PROMPT
    )

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "system",
                "content": self._system_prompt,
            },
            {
                "role": "user",
                "content": self._template.format(**input),
            },
        ]


class CleanNumberedList(Step):
    """A step to clean the numbered list of questions."""

    def process(self, inputs: StepInput) -> StepOutput:
        import re

        pattern = r"^\d+\.\s"

        for input in inputs:
            input["question"] = re.sub(pattern, "", input["question"])
        yield inputs
