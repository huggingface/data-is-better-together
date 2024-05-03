import os
import time
from typing import Any, Dict

import argilla as rg
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadHubDataset,
    StepInput,
    StepOutput,
    step,
)
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.steps.tasks.typing import ChatType
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login

from custom_preference_to_argilla import CustomPreferenceToArgilla

load_dotenv()

##################################
# Configuration
# This section contains the configuration for the pipeline.
# This is where you can define the model ID, the maximum number of new tokens to generate, the input batch size for the model via the Inference Endpoints API, and the Argilla configuration.
##################################


# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
MAX_NEW_TOKENS = 4000  # Maximum number of new tokens to generate

# Inference Endpoints Configuration
# INFERENCE_ENDPOINTS_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"  # Inference endpoints URL
# ENDPOINT_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
INPUT_BATCH_SIZE = 10  # Input batch size `for the model via the Inference Endpoints API, you can adjust this based on the model's requirements and the hardware you are using to deploy the model

# Argilla Configuration
ARGILLA_SPACE_URL = "https://dibt-demo-argilla-space.hf.space"  # Argilla Space URL
ARGILLA_DATASET_NAME = "aya_english_dpo"  # Argilla dataset name
ARGILLA_WORKSPACE_NAME = "admin"  # Argilla workspace name
# Dataset Configuration
INPUT_DATASET_HUB_ID = "DIBT/aya_dataset_english_example"  # Input dataset hub ID (created in the previous step)
OUTPUT_DATASET_HUB_ID = "DIBT/aya_english_dpo_raw"  # Output dataset hub ID
SPLIT = "train"  # Split of the dataset to use. Start with test whilst you are testing the pipeline and then switch to train when you are ready to generate the full dataset

HUGGINGFACE_TOKEN = os.getenv("HF_API_KEY")

#######################################
# Check required environment variables
#######################################
assert (
    HUGGINGFACE_TOKEN is not None
), "Please set the HF_API_KEY environment variable or authenticate with the Hugging Face CLI using `huggingface-cli login`"
login(token=HUGGINGFACE_TOKEN)
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")

# Check if the API key is set
assert (
    ARGILLA_API_KEY is not None
), "Please set the ARGILLA_API_KEY environment variable or pass it as a parameter"

#####################
# Helper functions
#####################


def remove_existing_dataset(argilla_dataset_name: str):
    """Remove an existing dataset from Argilla. This is useful when re-running the pipeline multiple times."""
    try:
        rg.init(
            api_url=ARGILLA_SPACE_URL,
            api_key=ARGILLA_API_KEY,
            workspace=ARGILLA_WORKSPACE_NAME,
        )
        argilla_ds = rg.FeedbackDataset.from_argilla(argilla_dataset_name)
        argilla_ds.delete()
    except ValueError as e:
        print(e)


#####################################
# Define distilabel custom steps
#####################################


@step(
    inputs=["generation"],
    outputs=["predicted_generation_language", "predicted_generation_language_score"],
)
def language_predict(inputs: StepInput) -> StepOutput:
    """
    A step to predict the language of the generated text.
    Sometimes models fail to generate text in the desired language.
    This step helps to identify such cases using an external language prediction model.
    """
    for input in inputs:
        try:
            cleaned_input = input["generation"].replace("\n", " ")
            resp = InferenceClient("laurievb/OpenLID").text_classification(
                cleaned_input
            )
            top_prediction = resp[0]  # top prediction is the first element in the list
            input["predicted_generation_language"] = top_prediction.label
            input["predicted_generation_language_score"] = min(
                1.0, top_prediction.score
            )  # ensure score is between 0 and 1
        except Exception as e:
            print(e)
            input["predicted_generation_language"] = "error"
            input["predicted_generation_language_score"] = 0.0
    yield inputs


@step(inputs=["targets", "generation"], outputs=["generations"])
def CombineAyaAndModelResponse(
    inputs: StepInput,
) -> StepOutput:
    """A step to combine the Aya and model responses and add the response sources."""
    for input in inputs:
        input["generations"] = [input["targets"], input["generation"]]
        input["generation_models"] = ["aya", MODEL_ID]
    yield inputs


#######################################################################
# Define a custom TextGeneration task focused on our target language
#######################################################################


# Custom system prompt
# This translates to something like:
# You are an AI assistant. Your primary language is Dutch. Answer most questions and prompts in Dutch, unless specifically asked to use another language.
# If you are asked to translate between two other languages, for example from French to English, perform the requested translation.
# When quotes or passages in another language are given in a prompt, assume that the user wants you to understand them and refer to them when formulating your English response. Do not translate the foreign text yourself, unless specifically requested.


# system_prompt = """Je bent een AI-assistent. Je primaire taal is Nederlands. Beantwoord de meeste vragen en prompts in het Nederlands, tenzij specifiek gevraagd wordt om een andere taal te gebruiken.
# Als je gevraagd wordt om te vertalen tussen twee andere talen, bijvoorbeeld van Frans naar Engels, voer dan de gevraagde vertaling uit. Wanneer citaten of passages in een andere taal in een prompt worden gegeven, ga er dan van uit dat de gebruiker wil dat je ze begrijpt en ernaar verwijst bij het formuleren van je Nederlandse antwoord. Vertaal de anderstalige tekst zelf niet, tenzij dit specifiek wordt gevraagd."""


# class DutchTextGeneration(TextGeneration):
#     """A TextGeneration task adds an additional system prompt."""

#     def format_input(self, input: Dict[str, Any]) -> "ChatType":
#         return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": input["instruction"]},
#         ]


#####################################
# Define the pipeline
#####################################

with Pipeline(name="generate-dpo-responses") as pipeline:
    # Load the dataset from the Hugging Face Hub
    load_hub_dataset = LoadHubDataset(
        name="load_dataset",
        output_mappings={"inputs": "instruction"},
    )
    #####################################
    # Define the LLM
    #####################################
    llm = InferenceEndpointsLLM(
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_display_name=MODEL_ID,
        api_key=HUGGINGFACE_TOKEN,
    )
    # Generate responses using the model
    text_generation = TextGeneration(
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
    to_argilla = CustomPreferenceToArgilla(
        name="to_argilla",
        api_url=ARGILLA_SPACE_URL,
        api_key=ARGILLA_API_KEY,
        dataset_name=ARGILLA_DATASET_NAME,
        dataset_workspace=ARGILLA_WORKSPACE_NAME,
        num_generations=2,
        metadata_properties=[
            rg.TermsMetadataProperty(name="predicted_generation_language").dict(),  # type: ignore
            rg.FloatMetadataProperty(  # type: ignore
                name="predicted_generation_language_score", min=0.0, max=1.0
            ).dict(),
        ],
    )
    ultrafeedback.connect(to_argilla)

#####################################
# Run the pipeline
#####################################

if __name__ == "__main__":
    start_time = time.time()
    if ARGILLA_DATASET_NAME:
        print(f"Removing existing dataset: {ARGILLA_DATASET_NAME}")
        remove_existing_dataset(ARGILLA_DATASET_NAME)
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
    dataset.push_to_hub(OUTPUT_DATASET_HUB_ID, token=HUGGINGFACE_TOKEN)
    end_time = time.time()
    print(f"Output dataset: https://huggingface.co/datasets/{OUTPUT_DATASET_HUB_ID}")
    print(f"Time taken: {end_time - start_time} seconds")
