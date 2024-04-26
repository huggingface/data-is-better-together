import os
import time
from typing import Any, Dict, List

import argilla as rg
from datasets import Dataset, load_dataset
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadHubDataset,
    PreferenceToArgilla,
    StepInput,
    StepOutput,
    step,
)
from distilabel.steps.tasks import TextGeneration
from distilabel.steps.tasks.text_generation import TextGeneration
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login
from distilabel.steps.tasks import TextGeneration, UltraFeedback

load_dotenv()

# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
MAX_NEW_TOKENS = 2000  # Maximum number of new tokens to generate

# Inference Endpoints Configuration
# INFERENCE_ENDPOINTS_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"  # Inference endpoints URL
# ENDPOINT_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
INPUT_BATCH_SIZE = 20  # Input batch size for the model via the Inference Endpoints API, you can adjust this based on the model's requirements and the hardware you are using to deploy the model

# Argilla Configuration
ARGILLA_SPACE_URL = "https://dibt-demo-argilla-space.hf.space"  # Argilla Space URL
ARGILLA_DATASET_NAME = "aya_dutch_dpo"  # Argilla dataset name
ARGILLA_WORKSPACE_NAME = "admin"  # Argilla workspace name
# Dataset Configuration
INPUT_DATASET_HUB_ID = "DIBT/aya_dataset_dutch_example"  # Input dataset hub ID (created in the previous step)
OUTPUT_DATASET_HUB_ID = "DIBT/aya_dutch_dpo_raw"  # Output dataset hub ID
SPLIT = (
    "test"  # Split of the dataset to use. We use testing whilst developing the pipeline
)

HUGGINGFACE_TOKEN = os.getenv("HF_API_KEY")
assert (
    HUGGINGFACE_TOKEN is not None
), "Please set the HF_API_KEY environment variable or authenticate with the Hugging Face CLI using `huggingface-cli login`"
login(token=HUGGINGFACE_TOKEN)
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")

# Check if the API key is set
assert (
    ARGILLA_API_KEY is not None
), "Please set the ARGILLA_API_KEY environment variable or pass it as a parameter"

# Initialize the Argilla client
rg.init(
    api_url=ARGILLA_SPACE_URL, api_key=ARGILLA_API_KEY, workspace=ARGILLA_WORKSPACE_NAME
)


def remove_existing_dataset(ARGILLA_DATASET_NAME: str):
    """Remove an existing dataset from Argilla. This is useful when re-running the pipeline."""
    try:
        argilla_ds = rg.FeedbackDataset.from_argilla(ARGILLA_DATASET_NAME)
        argilla_ds.delete()
    except ValueError as e:
        print(e)


@step(
    inputs=["generation"],
    outputs=["predicted_generation_language", "predicted_generation_language_score"],
)
def language_predict(inputs: StepInput) -> StepOutput:
    """
    A step to predict the language of the generated text.
    Sometimes models fail to generate text in the desired language.
    """
    for input in inputs:
        try:
            cleaned_input = input.replace("\n", " ")
            resp = InferenceClient("laurievb/OpenLID").text_classification(
                cleaned_input
            )
            top_prediction = resp[0]  # top prediction is the first element in the list
            input["predicted_generation_language"] = top_prediction.label
            input["predicted_generation_language_score"] = top_prediction.score
        except Exception as e:
            print(e)
            input["predicted_generation_language"] = "error"
            input["predicted_generation_language_score"] = 0.0
    yield inputs


# Define a step to combine the Aya and model responses and add the response sources
@step(inputs=["targets", "generation"], outputs=["generations"])
def CombineAyaAndModelResponse(
    inputs: StepInput,
) -> StepOutput:
    """A step to combine the Aya and model responses and add the response sources."""
    for input in inputs:
        input["generations"] = [input["targets"], input["generation"]]
        input["generation_models"] = ["aya", MODEL_ID]
    yield inputs


def update_argilla_dataset_with_metadata(
    dataset_name: str,
    workspace: str,
    hub_dataset: Dataset,
    metadata_keys: List[str] = None,
):
    """Update an Argilla dataset with metadata from the dataset."""
    # by default, we add the predicted generation language and response source
    if metadata_keys is None:
        metadata_keys = ["predicted_generation_language", "generation_models"]
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


system_prompt = """Je bent een AI-assistent. Je primaire taal is Nederlands. Beantwoord de meeste vragen en prompts in het Nederlands, tenzij specifiek gevraagd wordt om een andere taal te gebruiken.
Als je gevraagd wordt om te vertalen tussen twee andere talen, bijvoorbeeld van Frans naar Engels, voer dan de gevraagde vertaling uit. Wanneer citaten of passages in een andere taal in een prompt worden gegeven, ga er dan van uit dat de gebruiker wil dat je ze begrijpt en ernaar verwijst bij het formuleren van je Nederlandse antwoord. Vertaal de anderstalige tekst zelf niet, tenzij dit specifiek wordt gevraagd."""


class DutchTextGeneration(TextGeneration):
    """A TextGeneration task adds an additional system prompt."""

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input["instruction"]},
        ]


with Pipeline(name="generate-dpo-responses") as pipeline:
    # Load the dataset from the Hugging Face Hub
    load_hub_dataset = LoadHubDataset(
        name="load_dataset",
        output_mappings={"inputs": "instruction"},
    )
    llm = InferenceEndpointsLLM(
        model_id=MODEL_ID,
        # endpoint_name=ENDPOINT_NAME,
        tokenizer_id=MODEL_ID,
        model_display_name=MODEL_ID,
        api_key=HUGGINGFACE_TOKEN,
    )

    # Generate responses using the model
    text_generation = DutchTextGeneration(
        name="text_generation",
        llm=llm,
        input_batch_size=INPUT_BATCH_SIZE,
        output_mappings={"model_name": "generation_model"},
        num_generations=1,
    )
    load_hub_dataset.connect(text_generation)
    language_prediction = language_predict(name="language_prediction")
    text_generation.connect(language_prediction)
    combine_columns = CombineAyaAndModelResponse(
        name="combine_columns",
    )

    language_prediction.connect(combine_columns)
    ultrafeedback = UltraFeedback(
        name="ultrafeedback", aspect="overall-rating", llm=llm
    )
    combine_columns.connect(ultrafeedback)
    to_argilla = PreferenceToArgilla(
        name="to_argilla",
        dataset_name=ARGILLA_DATASET_NAME,
        api_url=ARGILLA_SPACE_URL,
        api_key=ARGILLA_API_KEY,
        dataset_workspace=ARGILLA_WORKSPACE_NAME,
        num_generations=2,
    )
    ultrafeedback.connect(to_argilla)

if __name__ == "__main__":
    # time the pipeline
    start_time = time.time()
    if ARGILLA_DATASET_NAME:
        print(f"Removing existing dataset: {ARGILLA_DATASET_NAME}")
        remove_existing_dataset(ARGILLA_DATASET_NAME)
    # run the pipeline
    dataset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": INPUT_DATASET_HUB_ID,
                "split": SPLIT,
            },
            "text_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": True,
                        "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                    }
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
            workspace=ARGILLA_WORKSPACE_NAME,
            hub_dataset=hub_dataset,
        )
    except ValueError as e:
        print(e)
    print(f"Time taken: {end_time - start_time} seconds")
