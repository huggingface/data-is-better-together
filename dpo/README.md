WIP!!

Roughly the goal is to boostrap more DPO datasets for different languages.

The [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) is a 

> a multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform from Cohere For AI. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators.

Rough steps:

- Start from [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset)
- Filter to a specific language
- Run the `aya_dpo_gen.py` script to generate a new response for each prompt
- Send to Argilla for annotation

There are different approaches to using the resulting dataset:
- Depending on the language, and the availability of strong models, you might always choose the Aya/human response over the generated one as the "chosen" response.
- For other languages with stronger LLMs sometimes spent annotating will help you understand the differences between the generated and human responses. 
- If you can gather a group of motivated annotators, you can create a true preference dataset where the annotators choose between the generated and human responses.

