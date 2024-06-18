<p align="center">
  <img src="https://huggingface.co/blog/assets/community-datasets/thumbnail.png" width="500px"/>
</p>

<p align="center">🤗 <a href="https://huggingface.co/DIBT" target="_blank">Spaces & Datasets</a></p>

# Data is Better Together

> If you are working on a valuable community-developed dataset but are limited by available resources, please reach out to us on the Hugging Face discord. We may be able to provide support to enhance your project.

Data is Better Together is a collaboration between 🤗 Hugging Face, 🏓 Argilla, and the Open-Source ML community. We aim to empower the open-source community to build impactful datasets collectively. This initiative consists of two main components: the community efforts and the cookbook efforts.

<details open>
  <summary><strong>Community Efforts</strong>: They were guided by the HF Team, hands-on projects focused on creating valuable datasets. These projects required the participation of the community and have been successfully completed.</summary>

  <ul>
  <details>
  <summary><strong>Prompt ranking</strong></summary>

  - **Goal**: This project aimed to create a dataset of 10k prompts ranked by quality. These prompts included both synthetic and human-generated from various datasets. The intention was to use the final dataset for prompt ranking tasks or synthetic data generation. You can find more information about this project in the [prompt ranking README](community-efforts/prompt_ranking/README.md)
  - **How**: First, we prepared a dataset with the prompts to be ranked using Argilla in a Hugging Face Space. Then, we invited the community to rank the prompts based on their quality. Finally, we collected the annotations and released the dataset.
  - **Result**: Over 385 people joined this initiative! Thanks to their contribution, we released [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked). This dataset can be used for different tasks as you can filter the higher-quality prompts (for instance, see the MPEP project) and generate the corresponding completions. You can also find some models built on top of it [here](https://huggingface.co/models?dataset=dataset:DIBT/10k_prompts_ranked).
  </details>
  

  <details>
  <summary><strong>Multilingual Prompt Evaluation Project (MPEP)</strong></summary>

  - **Goal**: There are not enough language-specific benchmarks for open LLMs! So, we wanted to create a leaderboard for more languages by leveraging the community. This way, we could evaluate the performance of models using [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval). You can find more information about this project in the [MPEP README](community-efforts/prompt_translation/README.md).
  - **How**: We selected a subset of 500 high-quality prompts from the [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) (see the prompt ranking project) and asked the community to help us translate this curated prompt dataset into different languages.
  - **Result**: We achieved to translate the whole dataset for Dutch and Russian, and almost finished with Spanish. Many other languages have also joined this initiative. You can take a look at the resulting datasets [here](https://huggingface.co/datasets?search=MPEP).
</details>
</ul>

<details open>
  <summary><strong>Cookbook Efforts</strong>: They aim to create guides and tools that help the community in building valuable datasets. They are not guided by the HF team and expected to be handled standalone, allowing you to freely contribute or use them to create your own unique dataset.</summary>

  <ul>
  <details>
  <summary><strong>Domain Specific Datasets</strong></summary>

  This project aims to bootstrap the creation of more domain-specific datasets for training models. The **goal** is to create a set of tools that help users to collaborate with domain experts. Find out more in the [Domain Specific Datasets README.](cookbook-efforts/domain-specific-datasets/README.md)
  </details>

  <details>
  <summary><strong>DPO/ORPO Datasets</strong></summary>

  Many languages do not have DPO datasets openly shared on the Hugging Face Hub. The [DIBT/preference_data_by_language](https://huggingface.co/spaces/DIBT/preference_data_by_language) Space gives you an overview of language coverage of DPO datasets for different languages. The **goal** of this project is to help foster a community of people building more DPO-style datasets for different languages. Find out more in this [DPO/ORPO datasets README](cookbook-efforts/dpo-orpo-preference/README.md).
</details>

  <details>
  <summary><strong>KTO Datasets</strong></summary>

  KTO is another type of preference dataset that can be used to train models to make decisions. Unlike DPO, it doesn't require two candidate responses. Instead, it relies on a simple binary preference, i.e. 👍👎. Thus, data is easier to collect and annotate. The **goal** of this project is to help the community create their own KTO dataset. Find out more in this [KTO datasets README](cookbook-efforts/kto-preference/README.md)

  </details>
  </ul>

**🤝​ How can I contribute to the cookbook efforts?** That's easy! You can contribute by following the instructions in the README of the project you are interested in. Then, share your results with the community!

</details>
