# Domain Specific Dataset Project

This is project to bootstrap the creation of domain-specific datasets for training models. The goal is to create a set of tools that help users to collaborate with domain experts.

## Example Farming Project

Here are examples of the resources for our farming example:
The seeding space: https://huggingface.co/spaces/argilla/domain-specific-seed for the domain expert.
The demo dataset: https://huggingface.co/datasets/argilla/farming we've created
The argilla space: https://huggingface.co/spaces/argilla/farming
The pipeline code: https://github.com/argilla-io/distilabel-workbench/tree/main/projects/farming

## Project Structure

- `app/` : A streamlit app to help domain experts to define seed data like system prompt and topics, by creating an empty dataset on the hub.
- `app/pipeline.py` : The distilabel pipeline code that is used to create the dataset.
- `scripts/` : Adhoc scripts that we used to ease annotation with vector search.