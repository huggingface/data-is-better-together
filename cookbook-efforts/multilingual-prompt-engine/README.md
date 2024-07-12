![mpe-logo](./assets/mpe-logo.png.jpg)

# Mulitlingual Prompt Engine (MPE) Project

MPE is a dataset creation tool that can be used to prompt defined LLMs with a dataset or datasets of your choosing and collect the responses.

## What is the goal of this project?

The goal of this project is to continue the [MPEP initiative](../../community-efforts/prompt_translation/README.md) using the multilingual prompt translations created by the community to prompt the [best open source LLMs available](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) and collect their responses. 

These responses are then:

1. Used to create novel datasets which contain both prompts and responses.
2. Uploaded to an [Argilla Space](https://huggingface.co/docs/hub/en/spaces-sdks-docker-argilla) for ranking by the community.

The latter effort will enable the creation of novel, language-specific performance metrics for multilingual LLMs and therefore directly contribute to the advancement of multilingual AI.

## Why do we need multilingual metrics and why does multilingual performance matter?

[Good AI requires good data](https://huggingface.co/blog/ethics-soc-6). Building meaningful multilingual benchmarks contributes to creating good datasets upon which foundation models may be trained to be performative in languages other than English.

Further to that point, the [well-known English-language bias](https://www.axios.com/2023/09/08/ai-language-gap-chatgpt) in AI means that [traditional metrics](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about) do not address model performance in handling prompts in languages other than English.

The MPEP initiative is taking two meaningful steps to address these issues:

1. Creating training data for models to handle prompts in languages other than English.
2. Creating a mechanism by which existing models may be scored based on their performance in specific languages other than English.

## How can you contribute?

Community members can contribute to the MPEP initiative (and MPE project) in a number of ways:

1. Contribute to the MPE codebase: Just open a PR with your changes and a short explanation of what you have done.
2. Contribute to the MPEP initiative as an annotator or language lead: Check out the [MPEP README](../../community-efforts/prompt_translation/README.md) for more information about which languages are already in progress, if you don't see your language, become a language lead, gather a group of annotators and build the dataset!
3. Contribute to the annotation of MPE results: The end product of the MPE project is an Argilla Space with each prompt and response across all languages and models. The annotating user identifies the language in which they would like to rank responses, and then offers a 1-5 ranking of the quality of the response to include judgements on quality of language use, syntax, vocabulary, grammar, and thoroughness of the response itself.

## Project Overview

This section of the documentation offers a configuration example of [`01_create_response_dataset.ipynb`](./01_create_response_dataset.ipynb) so that you may modify the code locally to prompt models of your choosing with datasets of your choosing as well.

### 1. Prerequisites

* A 🤗 Hugging Face account: We'll extensively use the Hugging Face Hub both to generate our data via hosted model APIs and to share our generated datasets. You can sign up for a Hugging Face account [here](https://huggingface.co/join).
* We presume that you already have known target datasets and a jury of models that you would like to use for this exercise if you plan to run the notebook locally.

### 2. Usage

Once you've cloned the repository locally and created a suitable virtual environment of your choosing, install the required libraries:

```bash
# This documentation presumes the user uses pip as their package manager, but ./environment.yml is also provided for conda users

$ pip install requirements.txt
```

Please create a `.env` environment variable file and include a [Hugging Face Token](https://huggingface.co/settings/tokens) with `write` permissions.

```txt
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXX"
```

Alternatively, you can hardcode your Hugging Face Token in the appropriate location of the notebook, but please ensure that you do not publish this notebook in a publically accessible location without obfuscating your token.

```python
hf_token = "hf_XXXXXXXXXXXXXXXXXXXXXXX"
```

Define the criteria by which you wish to search the Hugging Face Datasets Library. This notebook uses the `label` parameter. [This documentation](https://huggingface.co/docs/huggingface_hub/v0.23.4/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) offers guidance on modifying this parameter to meet your needs.

```python
mpep_datasets = search_datasets_by_label('MPEP')
```

Define the target models that you would like to prompt. The syntax is `{organization}/{model}`.

```python
# List of target models
models = [
    "google/gemma-2-27b-it",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "Qwen/Qwen2-72B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]
```

Define the number of prompts from the dataset to be used for each model. MPE intended to use all 500 prompts of any given dataset, but you may wish to modify this parameter to include only 10 prompts, chosen at random: 

```python
# Number of prompts to process
num_prompts = 5
```

Run the notebook. Your prompt / response pairs are written to the local `./responses` directory for your usage.

#### Hosted Model APIs

Due to the size of the most performative open source LLMs, we need to use dedicated Inference Endpoints for our models. Hugging Face has very generously extended non-trivial grants to this project in order to enable this capability.

If you are running this notebook locally, and depending on the size of the models that you wish to use, you may be able to use Hugging Face's free inference API to generate responses.

#### The dataset produced

Resultant data is written to the local `./responses` directory for each language. Asperationally, this notebook will be expanded to write data back into the target datasets or new, discrete datasets.

```python
# TO DO: Example result
```

### I'm GPU-poor, can I still get involved?

Of course! This script is published publically in the spirit of maintianing and continuing the fully open source MPEP initiative. The concepts demonstrated by this notebook are applicable in many use cases outside the scope of the goals pursued here. 

Community members can participate by:

1. [Annotating and ranking English-language prompts](https://huggingface.co/spaces/DIBT/prompt-collective).
2. [Translating the best prompts into a language of their choice](../../community-efforts/prompt_translation/README.md) as either annotators or language leads.
3. Annotating and ranking prompts in languages other than English to create novel multilingual metrics for popular LLMs (FORTHCOMING OPPORTUNITY)

## Next Steps

The notebook in its current state is incomplete with respect to the ultimate goal of the project, creating new or enhancing existing MPEP lanaguage datasets.

Further development is required to accomplish the following:

* Decide the best use of resultant data (update existing datasets or create new datasets).
* Enhance notebook code to perform the agreed-upon action.
* Creation of an Argilla Space for community annotators to assess the quality of created prompts.

## License

This software is available for use under the [Apache License Version 2.0](./LICENSE).
