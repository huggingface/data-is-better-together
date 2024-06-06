# Multilingual Prompt Evaluation Project (MPEP)

*🏅 There were not enough language-specific benchmarks for open LLMs. We wanted to create a leaderboard for more languages by leveraging the community!🏅*

## What is it?

The Multilingual Prompt Evaluation Project (MPEP) is a community-driven effort to evaluate the performance of open language models across different languages. We translated a curated set of 500 high-quality prompts into multiple languages with the aim of evaluating the performance of models in different languages using [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), an automated tool for evaluating instruction/chat models based on LLM evaluation.

## How did we make it possible?

As the community created a dataset of 10k prompts [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) with quality ratings as part of the Data is Better Together initiative. From this dataset, we curated a subset of 500 high-quality prompts that cover a diverse range of capabilities for a model, such as math, coding, relationships, email generation, etc.

However, these prompts were originally in English, so we asked the community to help us translate this curated dataset into different languages so that we could use the translated prompts to evaluate the performance of models for the languages we translate into.

## How did people contribute?

There were two ways to contribute to this effort: by becoming a language lead or as community contributor.

* The language leads were responsible for setting up a Hub organization and creating an Argilla Space for their language. They also gathered a community of people to help them translate the prompts and created a dashboard to track the progress of the translation effort with the guidance of Daniel van Strien. We need to thank them for their hard work!

* People who spoke the languages that were being translated into could contribute to the translation of prompts. They just needed a Hugging Face account to log in to the relevant Space and start translating the prompts.

## Which tools were used?

For the MPEP project, we used two main tools to help us manage the translation process.

- [Argilla](https://github.com/argilla-io/argilla): an open-source data annotation tool that we used for the translation of prompts. Argilla has the option of using Hugging Face for authentication, which makes it easier for the community to contribute to the translation of prompts.
- [Hugging Face Spaces](https://huggingface.co/spaces): a platform for hosting machine learning applications and demos. We'll use Spaces to host the Argilla tool for the translation of prompts.

To make easier the translation set up for users, we also created a series of notebooks that served as guidance.

## What did we achieve? <!--To be checked-->

We started efforts to translate the prompts into several languages (shown below). However, we could not complete all the translations. The successful ones were [Dutch](https://huggingface.co/datasets/DIBT/MPEP_DUTCH) and [Russian](https://huggingface.co/datasets/DIBT/MPEP_RUSSIAN), and almost finished with [Spanish](https://huggingface.co/datasets/DIBT/MPEP_SPANISH). However, many users showed interest in translating the prompts into other languages. You can take a look at the resulting datasets [here](https://huggingface.co/DIBT).

<table>
    <tr>
        <td>Dutch</td>
        <td>Russian</td>
        <td>Tagalog</td>
        <td>Spanish</td>
        <td>Malagasy</td>
    </tr>
    <tr>
        <td>Czech</td>
        <td>Arabic</td>
        <td>French</td>
        <td>Turkish</td>
        <td>German</td>
    </tr>
    <tr>
        <td>Vietnamese</td>
        <td>Portuguese</td>
        <td>Cantonese</td>
        <td>Slovak</td>
    </tr>
</table>