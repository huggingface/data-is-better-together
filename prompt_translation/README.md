 Multilingual Prompt Evaluation Project (MPEP)

*🏅 There are not enough language-specific benchmarks for open LLMs. We want to create a leaderboard for more languages by leveraging the community!🏅*

## How do we plan to do this?

The community has created a dataset of 10k prompts [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) with quality ratings as part of Data is Better Together.

From this dataset, we have curated a subset of 500 high-quality prompts that cover a diverse range of capabilities for a model, such as math, coding, relationships, email generation, etc.

We want to use these 500 prompts to evaluate the performance of models using [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) (an automated way of evaluating the performance of instruction/chat models).

Currently, our prompts are in English. We are asking the community to help us translate this curated prompt dataset into different languages so that we can use these translated prompts to evaluate the performance of models for the languages we translate into.

We are currently building a leaderboard to display the evaluation results as part of this. This will help language communities identify how open models perform across different languages.

## How do I contribute?

There are two ways to contribute to this effort:

- Become a language lead
- Contribute to the translation of prompts

### Become a language lead

This doesn’t mean you need to be fully responsible for everything! It does let us know that at least one person wants to work for a particular language. Once we know someone wants to work on a language task, we’ll help you create an annotation task for that language.

To nominate yourself as a language lead please join our [Discord channel](https://discord.gg/hugging-face-879548962464493619) and let us know which language you want to work on (please tag Daniel van Strien `@.dvs13`) and let us know your Hugging Face username.

#### Setup a Hub organization and create an Argilla Space for your language

For the MPEP project, we will use a few tools to help us manage the translation process.

- Argilla: an open-source data annotation tool that we'll use for the translation of prompts. Argilla has the option of using Hugging Face for authentication, which makes it easier for the community to contribute to the translation of prompts.
- Hugging Face Spaces is a platform for hosting machine learning applications and demos. We'll use Spaces to host the Argilla tool for the translation of prompts.

To get started, you will need to set up a Hub organization and create an Argilla Space for your language. We have created a series of notebooks to help you set up a Hub organization and create an Argilla Space for your language.

- This [notebook](./prompt_translation/01_setup_prompt_translation_space.ipynb) (<a href="https://colab.research.google.com/github/huggingface/data-is-better-together/blob/main/prompt_translation/prompt_translation/01_setup_prompt_translation_space.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>) will guide you through the process of setting up a Hub organization and creating an Argilla Space for your language.
- This [notebook](./prompt_translation/02_upload_prompt_translation_data.ipynb) (<a href="https://colab.research.google.com/github/huggingface/data-is-better-together/blob/main/prompt_translation/prompt_translation/02_upload_prompt_translation_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>) will guide you through the process of uploading the prompt translation data to your Argilla Space and optionally pre-translating the prompts using Hugging Face models. It will also show you how to create a dashboard to monitor the progress of the translation task!

#### Gather a community of people and begin translating!

Once an Argilla Space is created, anyone with a Hugging Face login can log in to an account and begin contributing to the translations. If no existing communities are focused on ML for your language, you may want to create a thread in the DIBT Discord channel to discuss with others. This might also lead to some excellent follow-up project ideas!

### Contribute to the translation of prompts

If there is an existing Argilla effort focused on a language you speak, you can contribute to the translation of prompts. You will just need a Hugging Face account to log in to the relevant Space. You can find the current active efforts [here](https://github.com/huggingface/data-is-better-together?tab=readme-ov-file#contribute-translations). The best way to keep up to date with everything that is happening is to join the [Discord channel](https://discord.gg/hugging-face-879548962464493619)

### Submit your translations

Once you have translated the prompts, ping us on Discord and we will help you submit the translations to the leaderboard.

### Glossary

#### Alpaca Eval

An automatic approach to evaluating instruction-following models. It uses LLM-based evaluation to make it less time-consuming and expensive to assess model performance. We'll use AlpacaEval to evaluate the performance of models using the translated prompts.
