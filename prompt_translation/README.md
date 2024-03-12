# Translation of 500 high-quality prompts for evaluating language models across languages

There are not enough language-specific benchmarks for open LLMs. We want to create a leaderboard for more languages by leveraging the community.

## How do we plan to do this?

The community has created a dataset of 10k prompts [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) ranked by quality as part of Data is Better Together.

From this dataset, we have curated a subset of 500 high-quality prompts that cover a diverse range of capabilities for a model, such as math, coding, relationships, email generation, etc.
We want to use these 500 prompts to evaluate the performance of models using AlpaceEval (an automated way of evaluating the performance of instruction/chat models).

Currently, our prompts are in English. We are asking the community to help us translate this curated prompt dataset into different languages so that we can use these translated prompts to evaluate the performance of models for the languages we translate into.
We will build a leaderboard to display the evaluation results as part of this. This will help language communities identify how open models perform across different languages.

## How do I contribute?

There are two ways to contribute to this effort:

- Become a language lead
- Contribute to the translation of prompts

### Become a language lead

This doesn’t mean you need to be fully responsible for everything! It does let us know that at least one person wants to work for a particular language. Once we know someone wants to work on a language task, we’ll help you create an annotation task for that language.

To nominate yourself as a language lead please join our [Discord channel](https://discord.gg/hugging-face-879548962464493619) and let us know which language you want to work on (please tag `@dvs13`) and let us know your Hugging Face username.

#### Setup a Hub organization and create an Argilla Space for your language

This [notebook](prompt_translation/setup_prompt_translation_space.ipynb) will guide you through the process of setting up a Hub organization and creating an Argilla Space for your language. Please don't hesitate to ask for help in the Discord channel if you need it.

#### Gather a community of people and begin translating!

Once an Argilla Space is created, anyone with a Hugging Face login can log in to an account and begin contributing to the translations. If no existing communities are focused on ML for your language, you may want to create a thread in the DIBT Discord channel to discuss with others. This might also lead to some excellent follow-up project ideas!

### Contribute to the translation of prompts

If there is an existing Argilla effort focused on a language you speak, you can contribute to the translation of prompts. You can find the current efforts we're working on in this GitHub issue TODO add link. 

### Submit your translations

Once you have translated the prompts, ping us on Discord and we will help you submit the translations to the leaderboard.

### Glossary

### Alpaca Eval

An automatic approach to evaluating instruction-following models. It uses LLM-based evaluation to make it less time-consuming and expensive to assess model performance. We'll use AlpacaEval to evaluate the performance of models using the translated prompts.