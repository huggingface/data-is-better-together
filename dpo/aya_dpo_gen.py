import os
import time
from typing import List

import argilla as rg
import yaml
from datasets import Dataset, load_dataset
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadHubDataset,
    PreferenceToArgilla,
    Step,
    StepInput,
    StepOutput,
    step,
)
from distilabel.steps.tasks import TextGeneration
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login

load_dotenv()


def load_config(file_path: str) -> dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config("nl_config.yaml")
MODEL_ID = config["MODEL_ID"]
INFERENCE_ENDPOINTS_URL = config["INFERENCE_ENDPOINTS_URL"]
INPUT_BATCH_SIZE = config["INPUT_BATCH_SIZE"]
ARGILLA_SPACE_URL = config["ARGILLA_SPACE_URL"]
INPUT_DATASET_HUB_ID = config["INPUT_DATASET_HUB_ID"]
OUTPUT_DATASET_HUB_ID = config["OUTPUT_DATASET_HUB_ID"]
MAX_NEW_TOKENS = config["MAX_NEW_TOKENS"]
SPLIT = config["SPLIT"]
ARGILLA_DATASET_NAME = config["ARGILLA_DATASET_NAME"]


HUGGINGFACE_TOKEN = os.getenv("HF_API_KEY")
login(token=HUGGINGFACE_TOKEN)
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")

assert (
    ARGILLA_API_KEY is not None
), "Please set the ARGILLA_API_KEY environment variable or pass it as a parameter"


rg.init(api_url=config["ARGILLA_SPACE_URL"], api_key=ARGILLA_API_KEY, workspace="admin")


def remove_existing_dataset(dataset_name: str):
    """Remove an existing dataset from Argilla."""
    try:
        argilla_ds = rg.FeedbackDataset.from_argilla(ARGILLA_DATASET_NAME)
        argilla_ds.delete()
    except ValueError as e:
        print(e)


# This step predicts the language of the generated text
# Sometimes models fail to generate text in the desired language
# This step can be used to help filter out such responses
class LanguagePredict(Step):
    def process(self, inputs: StepInput) -> StepOutput:
        """
        A step to predict the language of the generated text.
        Sometimes models fail to generate text in the desired language.
        """
        for input in inputs:
            try:
                cleaned_input = input["generation"].replace("\n", " ")
                resp = InferenceClient("laurievb/OpenLID").text_classification(
                    cleaned_input
                )
                top_prediction = resp[
                    0
                ]  # top prediction is the first element in the list
                input["predicted_generation_language"] = top_prediction.label
                input["predicted_generation_language_score"] = top_prediction.score
            except Exception as e:
                print(e)
                input["predicted_generation_language"] = "error"
                input["predicted_generation_language_score"] = 0.0
        yield inputs


@step(inputs=["targets", "generation"], outputs=["generations"])
def CombineAyaAndModelResponse(
    inputs: StepInput,
) -> StepOutput:
    """A step to merge the Aya and model responses"""
    for input in inputs:
        input["generations"] = [input["targets"], input["generation"]]
        input["response_source"] = ["aya", MODEL_ID]
    yield inputs


def update_argilla_dataset_with_metadata(
    dataset_name: str,
    workspace: str,
    hub_dataset: Dataset,
    metadata_keys: List[str] = None,
):
    if metadata_keys is None:
        metadata_keys = ["predicted_generation_language", "response_source"]
    dataset = rg.FeedbackDataset.from_argilla(
        dataset_name,
        workspace=workspace,
    )

    metadata_values = [hub_dataset[key] for key in metadata_keys]

    if any(len(values) != len(dataset.records) for values in metadata_values):
        raise ValueError(
            f"Number of metadata values does not match the number of records ({len(dataset.records)})"
        )

    modified_records = []
    for record, *metadata in zip(dataset.records, *metadata_values):
        for key, value in zip(metadata_keys, metadata):
            record.metadata[key] = value
        modified_records.append(record)

    dataset.update_records(modified_records)


with Pipeline(name="generate-dpo-responses") as pipeline:
    # Load the dataset from the Hugging Face Hub
    load_hub_dataset = LoadHubDataset(
        name="load_dataset",
        output_mappings={"inputs": "instruction"},
    )
    # Generate responses using the model
    text_generation = TextGeneration(
        name="text_generation",
        llm=InferenceEndpointsLLM(
            base_url=INFERENCE_ENDPOINTS_URL,
            tokenizer_id=MODEL_ID,
            model_display_name=MODEL_ID,
            api_key=HUGGINGFACE_TOKEN,
        ),
        input_batch_size=INPUT_BATCH_SIZE,
        output_mappings={"model_name": "generation_model"},
        num_generations=2,
    )
    load_hub_dataset.connect(text_generation)
    language_prediction = LanguagePredict(name="language_prediction")
    text_generation.connect(language_prediction)
    combine_columns = CombineAyaAndModelResponse(
        name="combine_columns",
    )
    language_prediction.connect(combine_columns)
    to_argilla = PreferenceToArgilla(
        name="to_argilla",
        dataset_name="aya_dpo_test6",
        api_url=ARGILLA_SPACE_URL,
        api_key=ARGILLA_API_KEY,
        dataset_workspace="admin",
        num_generations=2,
        input_mappings={
            "instruction": "instruction",
            "generations": "generations",
        },
    )
    combine_columns.connect(to_argilla)

if __name__ == "__main__":
    # time the pipeline
    start_time = time.time()
    if ARGILLA_DATASET_NAME:
        remove_existing_dataset(ARGILLA_DATASET_NAME)
    # run the pipeline
    dataset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": INPUT_DATASET_HUB_ID,
                "split": SPLIT,
            },
            "text_generation": {
                "generation_kwargs": {
                    "max_new_to`kens": MAX_NEW_TOKENS,
                    "return_full_text": False,
                }
            },
            "to_argilla": {"dataset_name": ARGILLA_DATASET_NAME},
        }
    )
    # push the dataset to the hub
    dataset.push_to_hub(OUTPUT_DATASET_HUB_ID, token=HUGGINGFACE_TOKEN)
    end_time = time.time()
    print(f"Output dataset: https://huggingface.co/datasets/{OUTPUT_DATASET_HUB_ID}")
    print("Updating Argilla dataset with extra metadata...")
    try:
        hub_dataset = load_dataset(OUTPUT_DATASET_HUB_ID, split="train")
        languages = hub_dataset["predicted_generation_language"]
        update_argilla_dataset_with_metadata(
            dataset_name=ARGILLA_DATASET_NAME,
            workspace="admin",
            hub_dataset=hub_dataset,
        )
    except ValueError as e:
        print(e)
    print(f"Time taken: {end_time - start_time} seconds")
