# How to generate a DPO/ORPO dataset for a new language?

Here is an overview of the steps we'll take to generate a DPO/ORPO dataset for a new language:

```mermaid
graph LR
A[Load Aya Dataset] --> B[Filter Aya to focus on target language]
B --> C[Generate new responses using Inference Endpoints]
C --> D[UltraFeedback is used to rank responses]
D --> E[Validate preferences using Argilla]
```

## Prerequisites

- [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset): a dataset of human-annotated prompt-completion pairs across 71 languages.
- [`distilabel`](https://github.com/argilla-io/distilabel/): a library for generating synthetic datasets.
- Hugging Face [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) for hosting models used to generate new responses.
- [Argilla](https://argilla.io/): a data collaboration tool which we can use to label preferences in the generated dataset.

## 1. Filter the Aya dataset to the language you are interested in

To make things a bit easier, we'll start by filtering the Aya dataset to focus on the language you are interested in. This will make it easier to generate new responses using inference endpoints. Aya includes data for 71 languages, so hopefully you'll be able to find the language you are interested in. If not, feel free to reach out on the Hugging Face Discord server to see if there are other resources available for your language and we can help you get started.

For this step you can use the [dataset prep](cookbook-efforts/dpo/01_data_prep.ipynb) notebook to filter the Aya dataset to the language you are interested in. This notebook also gives you a chance to explore the Aya dataset a little bit more and get a sense of the data you'll be working with.

## 2. Identify a strong base model for your language

Identify any existing strong instruction-tuned models for your language. Some languages may already have nice benchmarks and leaderboards that you can use to identify a strong base model. Discussing this with the community on the Hugging Face Discord server can also be helpful.

> [!TIP]
> If there are no existing benchmarks for your language, it is probably best to rely on vibe based checks initially. As part of the data generation pipeline for this project, you could also decide to generate multiple responses for each prompt. This will allow you to also generate some rankings for the responses and use these to identify a strong base model. Even if there are benchmarks available, the results of the vibe based checks can be useful especially if the benchmarks are not very robust.

### 2.1 Deploying the model on Hugging Face Inference Endpoints

In the Distilabel library pipeline we created for this project, we use Hugging Face [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) for hosting models used to generate new responses. This allow us to run the scripts without needing to have GPUs available locally. There are two main options for using the Inference Endpoints:

- serverless
- dedicated endpoints

The serverless option doesn't require any setup but is restricted to a smaller subset of models. This [blog post](https://huggingface.co/blog/inference-pro#supported-models) gives a good overview of the supported models.

> [!TIP]
> If you are able to use the serverless option, it is the easiest way to get started. You can find more information on how to use the serverless option in the [docs](https://huggingface.co/docs/inference-endpoints/index). You will have higher rate limits with a Pro account. If you are using a free account, you will have a lower rate limit.

The dedicated endpoints option allows a large number of the models on the Hugging Face Hub to be deployed. This option requires a bit more setup but is more flexible. You can find more information on how to set up an endpoint in the [docs](https://huggingface.co/docs/inference-endpoints/index).

> [!WARNING]
> Be aware that you will be charged for the use of the Inference Endpoints. You can find more information on the pricing in the [pricing docs](https://huggingface.co/docs/inference-endpoints/pricing). It is very strongly recommend to enable the autoscaling option when setting up the endpoint to avoid unnecessary charges.

> [!WARNING]
> If you have local GPUs available, you can also adapt this approach using other [inference frameworks](https://distilabel.argilla.io/latest/components-gallery/llms/) such as Ollama or vLLM.

## 3. Use `distilabel` to generate a second response for each prompt in the filtered Aya dataset

Once you have filtered the Aya dataset to the language you are interested in and identified a strong base model, you can use the `distilabel` library to generate a second response for each prompt in the filtered Aya dataset. This script will use the strong base model identified in step 2 to generate the second response. For this step it probably makes sense to run the script locally. Rather than on Google Colab or a similar platform, as you will be able to run the script for longer periods of time without timeouts.

### Setup

You should clone this repository and install the required dependencies. You can do this by running the following commands:

```bash
git clone https://github.com/huggingface/data-is-better-together
cd dpo
```

You should then create a virtual environment and install the required dependencies. I have migrated to using [`uv`](https://github.com/astral-sh/uv) for managing virtual environments. You can install `uv` and installing Python dependencies. You can use `uv` to create a virtual environment and install the required dependencies by running the following commands:

```bash
uv venv
# On macOS and Linux.
source .venv/bin/activate
# On Windows.
.venv\Scripts\activate
```

You can then install the required dependencies by running the following command:

```bash
uv pip install -r requirements.txt
```

> [!NOTE]
> You can use whatever method you prefer for managing virtual environments. If you are not familiar with `uv`, you can use `venv` or `conda` instead.

### Updating the configuration and system prompt

We will use the `distilabel` library to generate a second response for each prompt in the filtered Aya dataset. This script will use the strong base model identified in step 2 to generate the second response.

Use the `aya_dpo_gen.py` script to generate a second response for each prompt in the filtered Aya dataset. This script will use the strong base model identified in step 2 to generate the second response. This script is fairly heavily commented so you should be able to follow along with what is happening. I will highlight a few key points.

At the top of the script, there is a section for configuration. You should update the configuration to match your setup. Here is an example of the configuration section:

```python
# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
MAX_NEW_TOKENS = 2000  # Maximum number of new tokens to generate

# Inference Endpoints Configuration
# INFERENCE_ENDPOINTS_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"  # Inference endpoints URL
# ENDPOINT_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
INPUT_BATCH_SIZE = 10  # Input batch size `for the model via the Inference Endpoints API, you can adjust this based on the model's requirements and the hardware you are using to deploy the model

# Argilla Configuration
ARGILLA_SPACE_URL = "https://dibt-demo-argilla-space.hf.space"  # Argilla Space URL
ARGILLA_DATASET_NAME = "aya_dutch_dpo"  # Argilla dataset name
ARGILLA_WORKSPACE_NAME = "admin"  # Argilla workspace name
# Dataset Configuration
INPUT_DATASET_HUB_ID = "DIBT/aya_dataset_dutch_example"  # Input dataset hub ID (created in the previous step)
OUTPUT_DATASET_HUB_ID = "DIBT/aya_dutch_dpo_raw"  # Output dataset hub ID
SPLIT = "test"  # Split of the dataset to use. Start with test whilst you are testing the pipeline and then switch to train when you are ready to generate the full dataset
```

> [!WARNING]
If you have local GPUs available, you can also adapt this approach using other [inference frameworks](https://distilabel.argilla.io/latest/components-gallery/llms/) such as Ollama or vLLM.

If you are using the serverless option for the Inference Endpoints, you only need to pass the model ID. If you are using the dedicated endpoints option, you will need to pass the Inference Endpoints URL and the endpoint name.

For testing purposes, you can set the split to `test`. This will allow you to test the pipeline with a smaller subset of the data. Once you are happy with the pipeline, you can switch to `train` to generate the full dataset.

#### Custom system prompt

Depending on the model you are using to generate the second response, you may need to provide a custom system prompt. For the Dutch example, I found that using Llama 3 without a system prompt was generating responses that were almost always in English. I added a custom system prompt to the script to steer the model to answering in Dutch. Here is an example of the custom system prompt I used for the Dutch example:

```python
system_prompt = """Je bent een AI-assistent. Je primaire taal is Nederlands. Beantwoord de meeste vragen en prompts in het Nederlands, tenzij specifiek gevraagd wordt om een andere taal te gebruiken.
Als je gevraagd wordt om te vertalen tussen twee andere talen, bijvoorbeeld van Frans naar Engels, voer dan de gevraagde vertaling uit. Wanneer citaten of passages in een andere taal in een prompt worden gegeven, ga er dan van uit dat de gebruiker wil dat je ze begrijpt en ernaar verwijst bij het formuleren van je Nederlandse antwoord. Vertaal de anderstalige tekst zelf niet, tenzij dit specifiek wordt gevraagd."""

```

This translates to something like

> You are an AI assistant. Your primary language is Dutch. Answer most questions and prompts in Dutch, unless specifically asked to use another language.
> If you are asked to translate between two other languages, for example from French to English, perform the requested translation.
> When quotes or passages in another language are given in a prompt, assume that the user wants you to understand them and refer to them when formulating your English response. Do not translate the foreign text yourself, unless specifically requested.

This custom system prompt is passed to the model along with the prompt and the model generates a response in Dutch. When you are setting up the pipeline for your language, you should test the responses generated by the model to ensure they are in the correct language. This is an area where some experimentation with the system prompt may be necessary. Running the script with a small subset of the data will allow you to test the responses and adjust the system prompt as needed. Once you are happy with the responses generated by the model, you can run the script with the full dataset.

> [!TIP]
> Whilst it might be tempting to use a system prompt that tells the model to always respond in the target language there are prompts that sometimes require the model to respond in a different language. For example, in the Dutch dataset there are prompts that ask, in Dutch, for a bit of French text to be translated into English. Truly expecting a multilingual model to respond in the target language is a good way to get a sense of how well the model can handle the task.

#### Running the script

Since we are running on Inference Endpoints, the script will use the Hugging Face Inference API to generate the responses. This means most of the heavy computation is happening on the Hugging Face servers and you don't need to have a powerful machine to run the script.

You can run the script by running the following command:

```bash
python aya_dpo_gen.py
```

The script will generate a second response for each prompt in the filtered Aya dataset. The responses will be saved to a Huggingface dataset at the end of the run. The script will also generate an LLM ranking for each response. It is an open question how well LLMs will do as judges for non English languages. For Dutch the LLM rankings seem pretty good using LLama 3. If after doing a few runs it seems like the LLM rankings are not very good, you could remove this step from the pipeline.

> [!TIP]
> Whilst the LLM judge may not be very good it may still be worth keeping that in the pipeline. When you annotate the data in Argilla you can use the LLM ranking as a starting point and assign a human ranking. This will also give you a sense of how well the LLM judge is doing. This alone could be a valuable contribution to the community.

## 4. Send the generated dataset to Argilla for annotation

The distilabel pipeline will by default get pushed to the Hugging Face Hub. You will be able to do the following in the Argilla interface:

- View the generated responses
- Rate the responses
- Provide feedback on the responses
- Add your own responses (for example to slightly adjust the response generated by the model)

Additionally the interface allows you to filter by the following:

- the predicted language of the response
- the LLM ranking of each response

This will allow you to quickly filter the responses and focus on the ones that are most interesting to you.

> [!TIP]
> If the LLM judge is doing a good job you may choose to only annotate a subset of the responses. If the LLM judge is not doing a good job you may choose to annotate all of the responses. This will give you a sense of how well the LLM judge is doing and how much work is required to get a good dataset. If you setup Argilla with HF OAuth enabled you can also share the dataset with the community and get help with the annotation!

This video shows some of the features available in the Argilla interface:

https://github.com/huggingface/data-is-better-together/assets/8995957/f597ee36-a15a-4746-873c-6e13f97db5de

## 5. Load the annotated dataset and generate the final DPO/ORPO dataset

The [02_load_from_argilla.ipynb](./02_load_from_argilla.ipynb) notebook walks through the process of loading the annotated dataset from Argilla and generating the final DPO/ORPO dataset. This notebook will also give you a chance to explore the annotated dataset and get a sense of the preferences that have been assigned to each response. You will likely want to adapt this notebook to your use case but it should provide a good starting point.

## (optional) Share your scripts with the community

To help others generate DPO/ORPO datasets for more languages, you can share your scripts with the community. You can do this by creating a pull request to the overall [README](./README.md) for this project. You can already see some examples of how this has been done for the Dutch example. You can either directly add your scripts, notebooks etc to a subfolder of the `dpo` folder or you can add a link to a repository where you have stored your scripts.

## (optional) Train a model using the generated DPO/ORPO dataset and push forward the state of the art in your language 🚀

[The Alignment Handbook](https://github.com/huggingface/alignment-handbook) has a great guide on how to train a model using a DPO/ORPO dataset.

Alternatively, if you prefer to take a more hands of approach you could try using [AutoTrain](https://github.com/huggingface/autotrain-advanced) to train a model using the generated DPO/ORPO dataset. AutoTrain is a tool that allows you to train a model using a dataset and a configuration file.
