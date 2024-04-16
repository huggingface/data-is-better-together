import os
from typing import List

from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps.expand import ExpandColumns
from distilabel.steps.keep import KeepColumns
from distilabel.steps.tasks.self_instruct import SelfInstruct
from distilabel.steps.tasks.evol_instruct.base import EvolInstruct
from distilabel.llms.mistral import MistralLLM
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


def serialize_pipeline(
    argilla_api_key: str,
    argilla_api_url: str,
    argilla_dataset_name: str,
    topics: List[str],
    perspectives: List[str],
    domain_expert_prompt: str,
    pipeline_config_path: str = "pipeline.yaml",
):
    terms = create_topics(topics, perspectives)
    with Pipeline("farming") as pipeline:
        load_data = LoadDataFromDicts(
            name="load_data",
            data=[{"input": term} for term in terms],
            batch_size=64,
        )
        base_llm = MistralLLM(
            model="mistral-medium", api_key=os.getenv("MISTRAL_API_KEY")
        )
        expert_llm = MistralLLM(
            model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY")
        )

        self_instruct = SelfInstruct(
            name="self-instruct",
            application_description=APPLICATION_DESCRIPTION,
            num_instructions=5,
            input_batch_size=8,
            llm=base_llm,
        )

        evol_instruction_complexity = EvolInstruct(
            name="evol_instruction_complexity",
            llm=base_llm,
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
            llm=expert_llm,
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

    pipeline.save(path=pipeline_config_path, overwrite=True, format="yaml")
