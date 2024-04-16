from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadHubDataset,
)
from distilabel.steps.tasks import TextGeneration, ConversationTemplate
from huggingface_hub import get_token
import time

INFERENCE_ENDPOINTS_URL = (
    "https://fqk8v1jpa972cklj.us-east-1.aws.endpoints.huggingface.cloud"
)
MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"
INPUT_DATASET_HUB_ID = "davanstrien/aya_dataset_dutch"
OUTPUT_DATASET_HUB_ID = "davanstrien/aya_dpo"
MAX_NEW_TOKENS = 1024
ENDPOINT_NAME = "solar-10-7b-instruct-v1-0-bli"
INPUT_BATCH_SIZE = 10

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
            api_key=get_token(),
        ),
        input_batch_size=INPUT_BATCH_SIZE,
        output_mappings={"model_name": "generation_model"},
    )
    load_hub_dataset.connect(text_generation)


if __name__ == "__main__":
    # run the pipeline
    # start time
    start_time = time.time()
    dataset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": INPUT_DATASET_HUB_ID,
                "split": "train",
            },
            "text_generation": {
                "generation_kwargs": {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "return_full_text": False,
                },
            },
        }
    )
    dataset.push_to_hub(OUTPUT_DATASET_HUB_ID)
    # end time
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
