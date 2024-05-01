# How to generate a DPO/ORPO dataset for a new language?

An overview of the steps we'll take to generate a DPO/ORPO dataset for a new language:

```mermaid
graph LR
A[Load Aya Dataset] --> B[Filter Aya to focus on target language]
B --> C[Generate new responses using Inference Endpoints]
C --> D[UltraFeedback is used to rank responses]
D --> E[Validate preferences using Argilla]
```

## The resources and tools we'll be using

- [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset): a dataset of human-annotated prompt-completion pairs across 71 languages.
- [`distilabel`](https://github.com/argilla-io/distilabel/): a library for generating synthetic datasets.
- Hugging Face [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) for hosting models used to generate new responses.
- [Argilla](https://argilla.io/): a data collaboration tool which we can use to label preferences in the generated dataset.

## 1. Filter the Aya dataset to the language you are interested in

To make things a bit easier, we'll start by filtering the Aya dataset to focus on the language you are interested in. This will make it easier to generate new responses using inference endpoints. Aya includes data for 71 languages, so hopefully you'll be able to find the language you are interested in. If not, feel free to reach out on the Hugging Face Discord server to see if there are other resources available for your language and we can help you get started.

For this step you can use the [dataset prep](./01_datasets_prep.ipynb) notebook to filter the Aya dataset to the language you are interested in. This notebook also gives you a chance to explore the Aya dataset a little bit more and get a sense of the data you'll be working with.

## 2. Identify a strong base model for your language

Identify any existing strong instruction-tuned models for your language. Some languages may already have nice benchmarks and leaderboards that you can use to identify a strong base model. Discussing this with the community on the Hugging Face Discord server can also be helpful.

### 2.1 Deploying the model on Hugging Face Inference Endpoints

In the Distilabel library pipeline we created for this project, we use Hugging Face [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) for hosting models used to generate new responses. This allow us to run the scripts without needing to have GPUs available locally. There are two main options for using the Inference Endpoints:

- serverless
- dedicated endpoints

The serverless option doesn't require any setup but is restricted to a smaller subset of models. This [blog post](https://huggingface.co/blog/inference-pro#supported-models) gives a good overview of the supported models. 

The dedicated endpoints option allows a large number of the models on the Hugging Face Hub to be deployed. This option requires a bit more setup but is more flexible. You can find more information on how to set up an endpoint in the [docs](https://huggingface.co/docs/inference-endpoints/index). We recommend you consider the following when setting up an endpoint:

- Scaling to zero after a period of inactivity. This can be defined in the settings of the endpoint when you are creating it or updated later. This will ensure you are not charged for the endpoint when it is not in use.
- Some models will require a [custom inference handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler). This won't be necessary for many models but is something to keep in mind.

## 3. Use `distilabel` to generate a second response for each prompt in the filtered Aya dataset

Use the `aya_dpo_gen.py` script to generate a second response for each prompt in the filtered Aya dataset. This script will use the strong base model identified in step 2 to generate the second response.

Currently this script uses a config file to define some settings. You will find an example config file `nl_config.yml` that is used to generate a second response for Dutch. You can create a new config file for your language by copying the example config file and updating the settings as needed.

## 4. (Optional) Send the generated dataset to Argilla for annotation

The community can then choose which response is better for each prompt. This step is optional but can help to improve the quality of the dataset. The script `aya_dpo_gen.py` will take card of this step for you but it assumes you have an Argilla instance running already.
