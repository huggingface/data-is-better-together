<p align="center">
  <img src="https://huggingface.co/blog/assets/community-datasets/thumbnail.png" width="500px"/>
</p>

<p align="center">🤗 <a href="https://huggingface.co/DIBT" target="_blank">Spaces & Datasets</a></p>

# Data is Better Together

Data is Better Together is a collaboration between 🤗 Hugging Face, 🏓 Argilla, and the Open-Source ML community. We aim to empower the open-source community to build impactful datasets collectively. 

## What have we done so far?

The community has created a dataset of 10k prompts [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) ranked by quality as part of Data is Better Together.

## What are we currently working on?

We are working on several strands of work. Here are current active projects.

### 1. Prompt ranking

Our first DIBT activity is focused on ranking the quality of prompts. We have already released version 1.0 of this dataset [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked). So far, over 385 people have contributed annotations to this dataset, but we are continuing to collect more annotations!

- Follow the progress of this effort in this [dashboard](https://huggingface.co/spaces/DIBT/prompt-collective-dashboard)
- You can contribute to the ranking of prompts [here](https://huggingface.co/spaces/DIBT/prompt-collective)

### 2. Multilingual Prompt Evaluation Project (MPEP)

There are not enough language-specific benchmarks for open LLMs! We want to create a leaderboard for more languages by leveraging the community! You can find more information about this project in the [MPEP README](prompt_translation/README.md).

#### Contribute translations

*Want to contribute translations?* Currently, these translation efforts are underway:

- [Dutch](https://dibt-dutch-prompt-translation-for-dutch.hf.space)
- [Russian](https://dibt-russian-prompt-translation-for-russian.hf.space)
- [Tagalog](https://dibt-filipino-prompt-translation-for-filipino.hf.space/)
- [Spanish](https://somosnlp-dibt-prompt-translation-for-es.hf.space/)
- [Malagasy](https://dibt-malagasy-prompt-translation-for-malagasy.hf.space/)
- [Czech](https://dibt-czech-prompt-translation-for-czech.hf.space/)
- [Arabic](https://2a2i-prompt-translation-for-arabic.hf.space/)
- [French](https://dibt-french-prompt-translation-for-french.hf.space/)
- [Turkish](https://dibt-turkish-prompt-translation-for-turkish.hf.space/)
- [German](https://dibt-german-prompt-translation-for-german.hf.space)
- [Vietnamese](https://ai-vietnam-prompt-translation-for-vie.hf.space/)
- [Portuguese](https://dibt-portuguese-prompt-translation-for-portuguese.hf.space)
- [Cantonese](https://dibt-cantonese-prompt-translation-for-cantonese.hf.space/)
- [Slovak](https://dibt-slovak-prompt-translation-for-slovak.hf.space/)

*Want to work on a language that's not listed?* You can follow the steps to set up a new annotation effort by going to `prompt_translation/` and checking out the three notebooks:

- In the first one, you'll learn how to set up a prompt translation space using Argilla and Hugging Face Spaces.
- In the second one, you'll see how to upload the prompt translation data for the language of your choice.
- In the third one, we show how to set up a dashboard to track the annotation efforts easily.


## Other guides

The Data is Better Together community has created several guides to support efforts to create valuable datasets via the open source community. Currently, we have the following guides:

- [Creating a KTO preference dataset](kto-preference/README.md)
