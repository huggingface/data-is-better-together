from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadHubDataset,
)
from distilabel.steps.tasks import TextGeneration
from huggingface_hub import get_token
import time
from distilabel.steps import StepInput, StepOutput, Step
from huggingface_hub import InferenceClient

INFERENCE_ENDPOINTS_URL = (
    "https://fqk8v1jpa972cklj.us-east-1.aws.endpoints.huggingface.cloud"
)
MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"
INPUT_DATASET_HUB_ID = "davanstrien/aya_dataset_dutch"
OUTPUT_DATASET_HUB_ID = "davanstrien/aya_dpo"
MAX_NEW_TOKENS = 1024
ENDPOINT_NAME = "solar-10-7b-instruct-v1-0-bli"
INPUT_BATCH_SIZE = 10


class LanguagePredict(Step):
    def process(self, inputs: StepInput) -> StepOutput:
        """A step to predict the language of the generated text"""
        for input in inputs:
            try:
                resp = InferenceClient("laurievb/OpenLID").text_classification(
                    input["generation"]
                )
                top_prediction = resp["predictions"][0]
                input["predicted_generation_language"] = top_prediction.label
                input["predicted_generation_language_score"] = top_prediction.score
            except Exception as e:
                print(e)
                input["predicted_generation_language"] = "error"
                input["predicted_generation_language_score"] = 0.0
        yield inputs


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
    language_prediction = LanguagePredict(name="language_prediction")
    text_generation.connect(language_prediction)


if __name__ == "__main__":
    # run the pipeline
    # start time
    start_time = time.time()
    dataset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": INPUT_DATASET_HUB_ID,
                "split": "test",
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
