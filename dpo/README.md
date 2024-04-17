# Generating DPO/ORPO datasets for more languages

Currently, many languages do not have DPO datasets openly shared. The goal of this project is to help foster a community of people building more DPO datasets for different languages.

The tl;dr of what we're doing is:

- start from [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) a multilingual instruction fine-tuning dataset that contains a total of 204k human-annotated prompt-completion pairs across TODO languages.
- filter to a specific language you want to generate a DPO dataset for.
- use `distilabel` to generate a second response for each prompt in the filtered Aya dataset.
- (optional) send the generated dataset to Argilla for annotation where the community can choose which response is better.

## What is Direct Preference Optimization (DPO/ORPO)?

Direct Preference Optimization (DPO) is a technique for training models to optimize for human preferences.

> [Direct Preference Optimization (DPO)](https://huggingface.co/papers/2305.18290) has emerged as a promising alternative for aligning Large Language Models (LLMs) to human or AI preferences. Unlike [traditional alignment methods](https://huggingface.co/blog/rlhf), which are based on reinforcement learning, DPO recasts the alignment formulation as a simple loss function that can be optimised directly on a dataset of preferences ${(x, y_w, y_l)}$, where $x$ is a prompt and $(y_w,y_l)$ are the preferred and dispreferred responses.  [source](https://huggingface.co/blog/pref-tuning)

Or, in other words, to train a model using DPO you need a dataset of prompts and responses where one response is preferred over the other.

![Sample of a preference tuning dataset.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/data.png)
*Example of a preference tuning dataset. Each row contains a prompt and a "chosen" and "rejected" response.*

## Why do we need DPO/ORPO datasets for more languages?

DPO datasets are a powerful tool for fine-tuning language models to generate responses that are more aligned with human preferences, so are a valuable resource for improving the quality of chatbots and other generative models. However, currently, there are only a few DPO datasets available for a limited number of languages. By generating more DPO datasets for different languages, we can help to improve the quality of generative models in a wider range of languages.

Recently, (Odds Ratio Preference Optimization) ORPO has been proposed as an alternative to DPO. ORPO is a novel approach to fine-tuning language models that incorporates preference alignment directly into the supervised fine-tuning (SFT) process by using the odds ratio to contrast favored and disfavored generation styles. By applying a minor penalty to the disfavored style during SFT, ORPO effectively guides the model toward the desired behavior without the need for an additional alignment step.

*tl;dr*: if you have a DPO-style dataset + a strong base model you can use ORPO to train a chat model. Recently Argilla, KAIST, and Hugging Face used this approach to train [HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1](https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1) a very strong chat model using only 7k data preference pairs!

## How can I get involved?

As part of Data Is Better Together, we're supporting the community in generating more DPO/ORPO datasets for different languages. If you would like to help, you can follow the steps below to generate a DPO/ORPO dataset for a language that you are interested in. There are already many language communities working together on the Hugging Face Discord server, so you can also join the server to collaborate with others on this project 🤗.

## How to generate a DPO/ORPO dataset for a new language?

An overview of the steps we'll take to generate a DPO/ORPO dataset for a new language:

1. **Filter the Aya dataset to the language you are interested in** using the [dataset prep](./01_datasets_prep.ipynb) notebook. This notebook also gives you a chance to explore the Aya dataset a little bit more. 
2. Identify any existing strong instruction-tuned models for your language. Some languages may already have nice benchmarks and leaderboards that you can use to identify a strong base model. Discussing this with the community on the Hugging Face Discord server can also be helpful.
3. **Use `distilabel` to generate a second response for each prompt in the filtered Aya dataset**. We'll use a script `aya_dpo_gen.py` to generate a second response for each prompt in the filtered Aya dataset. This script will use a strong base model to generate the second response.
4. **(Optional) Send the generated dataset to Argilla for annotation**. The community can then choose which response is better for each prompt. This step is optional but can help to improve the quality of the dataset.

### FAQs


#### I'm GPU poor

Inference endpoints + serverless (some languages might be able to use free inference endpoints)
