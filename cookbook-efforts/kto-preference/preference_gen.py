import os
import random

from datasets import load_dataset
from distilabel.llm import LLM, InferenceEndpointsLLM, LLMPool, ProcessLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Task, TextGenerationTask
from dotenv import load_dotenv

load_dotenv()

# You need to set the HF_TOKEN environment variable to your Hugging Face API token
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "Please set HF_TOKEN to your Hugging Face API token"
HF_USER_NAME = None
assert HF_USER_NAME, "Please set HF_USER_NAME to your Hugging Face username"

# if you want to sample from the dataset, set this to the number of samples you want
# if the size of your sample is larger than the dataset the full dataset will be used
SAMPLE_SIZE = None


## Load the dataset of prompts
def prepare_data():
    prompts = load_dataset("davanstrien/haiku_prompts", split="train")
    print(f"Loaded {len(prompts)} prompts")
    return prompts.rename_column("instructions", "input")


dataset = prepare_data()

## Define the task

task = TextGenerationTask(
    system_prompt="""You are a poet specialising in creating Haiku. \nYour haiku consist of three lines, with five syllables in the first line, seven in the second, and five in the third.\nBeyond being technically correct, your haiku should also be beautiful and meaningful. \nYou respond only with a haiku. You do not add anything else to your responses. \n\n""",
)

print(task.system_prompt)


# load llms
def load_llama2(task: Task) -> LLM:
    return InferenceEndpointsLLM(
        "meta-llama/Llama-2-70b-chat-hf",
        token=HF_TOKEN,
        task=task,
        max_new_tokens=512,
        prompt_format="llama2",
    )


def load_mistral(task: Task) -> LLM:
    checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
    return InferenceEndpointsLLM(
        checkpoint,
        token=HF_TOKEN,
        task=task,
        max_new_tokens=512,
        prompt_format="llama2",
    )


# uncomment to use nous-hermes-2-yi-34b-aug

# def load_nous_yi(task: Task) -> LLM:
#     checkpoint = "nous-hermes-2-yi-34b-aug"
#     return InferenceEndpointsLLM(
#         checkpoint,
#         token=HF_TOKEN,
#         task=task,
#         max_new_tokens=488,
#         prompt_format="chatml",
#     )


mistral = ProcessLLM(task=task, load_llm_fn=load_mistral)
llama2 = ProcessLLM(task=task, load_llm_fn=load_llama2)
# uncomment to use nous-hermes-2-yi-34b-aug
# nous_yi = ProcessLLM(task=task, load_llm_fn=load_nous_yi)

llms = [
    mistral,
    llama2,
]  # nous_yi] # uncomment to use nous-hermes-2-yi-34b-aug


pool = LLMPool(llms=llms)


pipeline = Pipeline(generator=pool)

if SAMPLE_SIZE is not None:
    sample_idx = random.sample(range(len(dataset)), min(SAMPLE_SIZE, len(dataset)))
    dataset = dataset.select(sample_idx)
print(f"Using {len(dataset)} prompts")

print("Generating haiku...")
haiku = pipeline.generate(
    dataset,
    num_generations=3,
    batch_size=1,
    display_progress_bar=True,
    shuffle_before_labelling=False,
)

print(haiku)
print("Pushing to hub...")
haiku.push_to_hub(f"{HF_USER_NAME}/haiku_dpo", "aesthetic-preference", token=HF_TOKEN)
