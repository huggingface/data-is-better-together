import json
from textwrap import dedent
from typing import Any, Dict, List

from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import TextGenerationToArgilla
from distilabel.steps.expand import ExpandColumns
from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps.tasks.self_instruct import SelfInstruct
from distilabel.steps.tasks.text_generation import TextGeneration
from distilabel.steps.tasks.typing import ChatType


################################################################################
# Functions to create task prompts
################################################################################


def create_application_instruction(domain: str, examples: List[Dict[str, str]]):
    """Create the instruction for Self-Instruct task."""
    system_prompt = dedent(
        f"""You are an AI assistant than generates queries around the domain of {domain}.
            Your should not expect basic but profound questions from your users.
            The queries should reflect a diversxamity of vision and economic positions and political positions.
            The queries may know about different methods of {domain}.
            The queries can be positioned politically, economically, socially, or practically.
            Also take into account the impact of diverse causes on diverse domains."""
    )
    for example in examples:
        question = example["question"]
        answer = example["answer"]
        system_prompt += f"""\n- Question: {question}\n- Answer: {answer}\n"""


def create_seed_terms(topics: List[str], perspectives: List[str]) -> List[str]:
    """Create seed terms for self intruct to start from."""

    return [
        f"{topic} from a {perspective} perspective"
        for topic in topics
        for perspective in perspectives
    ]


################################################################################
# Define out custom step for the domain expert
################################################################################


class DomainExpert(TextGeneration):
    """A customized task to generate text as a domain expert in the domain of farming and agriculture."""

    system_prompt: str
    template: str = """This is the the instruction: {instruction}"""

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.template.format(**input),
            },
        ]


################################################################################
# Main script to run the pipeline
################################################################################


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run the pipeline to generate domain-specific datasets."
    )
    parser.add_argument("--hub-token", type=str, help="The Hugging Face API token.")
    parser.add_argument("--argilla-api-key", type=str, help="The Argilla API key.")
    parser.add_argument("--argilla-api-url", type=str, help="The Argilla API URL.")
    parser.add_argument(
        "--argilla-dataset-name", type=str, help="The name of the dataset in Argilla."
    )
    parser.add_argument(
        "--seed_data_path",
        type=str,
        help="The path to the seed data.",
        default="seed_data.json",
    )
    parser.add_argument(
        "--endpoint-base-url", type=str, help="The base URL of the inference endpoint."
    )

    args = parser.parse_args()

    # collect our seed data

    with open(args.seed_data_path, "r") as f:
        seed_data = json.load(f)

    topics = seed_data.get("topics", [])
    perspectives = seed_data.get("perspectives", [])
    domain_expert_prompt = seed_data.get("domain_expert_prompt", "")
    examples = seed_data.get("examples", [])
    domain_name = seed_data.get("domain_name", "domain")

    # Define the task prompts

    terms = create_seed_terms(topics=topics, perspectives=perspectives)
    application_instruction = create_application_instruction(
        domain=domain_name, examples=examples
    )

    # Define the distilabel pipeline

    with Pipeline(domain_name) as pipeline:
        load_data = LoadDataFromDicts(
            name="load_data",
            data=[{"input": term} for term in terms],
            batch_size=64,
        )

        self_instruct = SelfInstruct(
            name="self_instruct",
            num_instructions=5,
            input_batch_size=8,
            llm=InferenceEndpointsLLM(
                base_url=args.endpoint_base_url,
                api_key=args.hub_token,
            ),
        )

        expand_instructions = ExpandColumns(
            name="expand_columns", columns={"instructions": "instruction"}
        )

        domain_expert = DomainExpert(
            name="domain_expert",
            llm=InferenceEndpointsLLM(
                base_url=args.endpoint_base_url,
                api_key=args.hub_token,
            ),
            input_batch_size=8,
            system_prompt=domain_expert_prompt,
        )

        to_argilla = TextGenerationToArgilla(
            name="text_generation_to_argilla",
            dataset_name=args.argilla_dataset_name,
            dataset_workspace="admin",
            api_url=args.argilla_api_url,
            api_key=args.argilla_api_key,
        )

        # Connect up the pipeline

        load_data.connect(self_instruct)
        self_instruct.connect(expand_instructions)
        expand_instructions.connect(domain_expert)
        domain_expert.connect(to_argilla)

    # Run the pipeline

    pipeline.run(
        parameters={
            "self_instruct": {
                "llm": {"api_key": args.hub_token, "base_url": args.endpoint_base_url}
            },
            "domain_expert": {
                "llm": {"api_key": args.hub_token, "base_url": args.endpoint_base_url}
            },
            "text_generation_to_argilla": {
                "dataset_name": args.argilla_dataset_name,
                "api_key": args.argilla_api_key,
                "api_url": args.argilla_api_url,
            },
        },
        use_cache=False,
    )
