import os
import subprocess
import time
from typing import List

from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps.expand import ExpandColumns
from distilabel.steps.keep import KeepColumns
from distilabel.steps.tasks.self_instruct import SelfInstruct
from distilabel.steps.tasks.evol_instruct.base import EvolInstruct
from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import TextGenerationToArgilla
from dotenv import load_dotenv

from domain import (
    DomainExpert,
    CleanNumberedList,
    create_topics,
    APPLICATION_DESCRIPTION,
)

load_dotenv()


def define_pipeline(
    argilla_api_key: str,
    argilla_api_url: str,
    argilla_dataset_name: str,
    topics: List[str],
    perspectives: List[str],
    domain_expert_prompt: str,
    hub_token: str,
    endpoint_base_url: str,
):
    """Define the pipeline for the specific domain."""

    terms = create_topics(topics, perspectives)
    with Pipeline("farming") as pipeline:
        load_data = LoadDataFromDicts(
            name="load_data",
            data=[{"input": term} for term in terms],
            batch_size=64,
        )
        llm = InferenceEndpointsLLM(
            base_url=endpoint_base_url,
            api_key=hub_token,
        )
        self_instruct = SelfInstruct(
            name="self-instruct",
            application_description=APPLICATION_DESCRIPTION,
            num_instructions=5,
            input_batch_size=8,
            llm=llm,
        )

        evol_instruction_complexity = EvolInstruct(
            name="evol_instruction_complexity",
            llm=llm,
            num_evolutions=2,
            store_evolutions=True,
            input_batch_size=8,
            include_original_instruction=True,
            input_mappings={"instruction": "question"},
        )

        expand_instructions = ExpandColumns(
            name="expand_columns", columns={"instructions": "question"}
        )
        cleaner = CleanNumberedList(name="clean_numbered_list")
        expand_evolutions = ExpandColumns(
            name="expand_columns_evolved",
            columns={"evolved_instructions": "evolved_questions"},
        )

        domain_expert = DomainExpert(
            name="domain_expert",
            llm=llm,
            input_batch_size=8,
            input_mappings={"instruction": "evolved_questions"},
            output_mappings={"generation": "domain_expert_answer"},
            _system_prompt=domain_expert_prompt,
        )

        keep_columns = KeepColumns(
            name="keep_columns",
            columns=["model_name", "evolved_questions", "domain_expert_answer"],
        )

        to_argilla = TextGenerationToArgilla(
            name="text_generation_to_argilla",
            dataset_name=argilla_dataset_name,
            dataset_workspace="admin",
            api_url=argilla_api_url,
            api_key=argilla_api_key,
            input_mappings={
                "instruction": "evolved_questions",
                "generation": "domain_expert_answer",
            },
        )

        load_data.connect(self_instruct)
        self_instruct.connect(expand_instructions)
        expand_instructions.connect(cleaner)
        cleaner.connect(evol_instruction_complexity)
        evol_instruction_complexity.connect(expand_evolutions)
        expand_evolutions.connect(domain_expert)
        domain_expert.connect(keep_columns)
        keep_columns.connect(to_argilla)
    return pipeline


def serialize_pipeline(
    argilla_api_key: str,
    argilla_api_url: str,
    argilla_dataset_name: str,
    topics: List[str],
    perspectives: List[str],
    domain_expert_prompt: str,
    hub_token: str,
    endpoint_base_url: str,
    pipeline_config_path: str = "pipeline.yaml",
):
    """Serialize the pipeline to a yaml file."""
    pipeline = define_pipeline(
        argilla_api_key=argilla_api_key,
        argilla_api_url=argilla_api_url,
        argilla_dataset_name=argilla_dataset_name,
        topics=topics,
        perspectives=perspectives,
        domain_expert_prompt=domain_expert_prompt,
        hub_token=hub_token,
        endpoint_base_url=endpoint_base_url,
    )
    pipeline.save(path=pipeline_config_path, overwrite=True, format="yaml")


def run_pipeline(
    pipeline_config_path: str = "pipeline.yaml",
    argilla_dataset_name: str = "domain_specific_datasets",
):
    """Run the pipeline and yield the output as a generator of logs."""

    command_to_run = [
        "python",
        "-m",
        "distilabel",
        "pipeline",
        "run",
        "--config",
        pipeline_config_path,
        "--param",
        f"text_generation_to_argilla.dataset_name={argilla_dataset_name}",
    ]

    # Run the script file
    process = subprocess.Popen(
        command_to_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    while process.stdout and process.stdout.readable():
        time.sleep(0.2)
        line = process.stdout.readline()
        if not line:
            break
        yield line.decode("utf-8")
