import json
from typing import Any, Dict

import argilla as rg
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromDicts,
    TextGenerationToArgilla,
    ExpandColumns,
)
from distilabel.steps.tasks import (
    TextGeneration,
    SelfInstruct,
)
from distilabel.steps.tasks.typing import ChatType
from huggingface_hub import hf_hub_download


################################################################################
# Define custom Argilla Dataset
################################################################################


def create_argilla_dataset(
    api_url: str,
    api_key: str,
    dataset_name: str,
    workspace: str,
):
    """Create a dataset in Argilla."""

    rg.init(api_url, api_key)
    rg_dataset = rg.FeedbackDataset(
        fields=[
            rg.TextField(name="id", title="id"),  # type: ignore
            rg.TextField(name="instruction", title="instruction"),  # type: ignore
            rg.TextField(name="generation", title="generation"),  # type: ignore
        ],
        questions=[
            rg.LabelQuestion(  # type: ignore
                name="quality",
                title=f"What's the quality of the generation for the given instruction?",
                labels={"bad": "👎", "good": "👍"},
            ),
            rg.TextQuestion(
                name="improved_instruction",
                title="How would you improve the instruction?",
                required=False,
            ),
            rg.TextQuestion(
                name="improved_response",
                title="How would you improve the response?",
                required=False,
            ),
        ],
    )
    try:
        rg_dataset.push_to_argilla(name=dataset_name, workspace=workspace)
    except RuntimeError as e:
        print(f"Failed to create the dataset in Argilla: {e} Moving on...")


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
    import os
    import json
    import sys

    # get some args
    repo_id = sys.argv[1]

    # Get super secret tokens

    hub_token = os.environ.get("HF_TOKEN")
    argilla_api_key = os.environ.get("ARGILLA_API_KEY", "owner.apikey")

    # load pipeline parameters

    with open(
        hf_hub_download(
            repo_id=repo_id, filename="pipeline_params.json", repo_type="dataset"
        ),
        "r",
    ) as f:
        params = json.load(f)

    argilla_api_url = params.get("argilla_api_url")
    argilla_dataset_name = params.get("argilla_dataset_name")
    self_instruct_base_url = params.get("self_instruct_base_url")
    domain_expert_base_url = params.get("domain_expert_base_url")
    self_intruct_num_generations = params.get("self_instruct_num_generations", 2)
    domain_expert_num_generations = params.get("domain_expert_num_generations", 2)
    self_instruct_temperature = params.get("self_instruct_temperature", 0.9)
    domain_expert_temperature = params.get("domain_expert_temperature", 0.9)
    self_instruct_max_new_tokens = params.get("self_instruct_max_new_tokens", 2048)
    domain_expert_max_new_tokens = params.get("domain_expert_max_new_tokens", 2048)

    if not all(
        [
            argilla_api_url,
            argilla_dataset_name,
            self_instruct_base_url,
            domain_expert_base_url,
        ]
    ):
        raise ValueError("Some of the pipeline parameters are missing")

    # collect our seed prompts defined in the space

    with open(
        hf_hub_download(
            repo_id=repo_id, filename="seed_data.json", repo_type="dataset"
        ),
        "r",
    ) as f:
        seed_data = json.load(f)

    application_instruction = seed_data.get("application_instruction")
    domain_expert_prompt = seed_data.get("domain_expert_prompt")
    domain_name = seed_data.get("domain")
    terms = seed_data.get("seed_terms")

    # Create the Argilla dataset

    create_argilla_dataset(
        api_url=argilla_api_url,
        api_key=argilla_api_key,
        dataset_name=argilla_dataset_name,
        workspace="admin",
    )

    # Define the distilabel pipeline

    with Pipeline(domain_name) as pipeline:
        load_data = LoadDataFromDicts(
            name="load_data",
            batch_size=64,
            data=[{"input": term} for term in terms],
        )

        self_instruct = SelfInstruct(
            name="self_instruct",
            num_instructions=self_intruct_num_generations,
            input_batch_size=8,
            llm=InferenceEndpointsLLM(
                api_key=hub_token,
                base_url=self_instruct_base_url,
            ),
            application_description=application_instruction,
        )

        expand_columns = ExpandColumns(
            name="expand_columns",
            columns=["instructions"],
            output_mappings={"instructions": "instruction"},
        )

        domain_expert = DomainExpert(
            name="domain_expert",
            llm=InferenceEndpointsLLM(
                api_key=hub_token,
                base_url=domain_expert_base_url,
            ),
            input_batch_size=8,
            num_generations=domain_expert_num_generations,
            system_prompt=domain_expert_prompt,
        )

        # Push the generated dataset to Argilla
        to_argilla = TextGenerationToArgilla(
            name="to_argilla",
            dataset_workspace="admin",
        )

        # Connect up the pipeline

        load_data.connect(self_instruct)
        self_instruct.connect(expand_columns)
        expand_columns.connect(domain_expert)
        domain_expert.connect(to_argilla)

    # Run the pipeline

    pipeline.run(
        parameters={
            "self_instruct": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": self_instruct_max_new_tokens,
                        "temperature": self_instruct_temperature,
                    },
                }
            },
            "domain_expert": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": self_instruct_max_new_tokens,
                        "temperature": domain_expert_temperature,
                    },
                }
            },
            "to_argilla": {
                "dataset_name": argilla_dataset_name,
                "api_key": argilla_api_key,
                "api_url": argilla_api_url,
            },
        },
        use_cache=False,
    )
