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

## Workflow for data generation

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="800" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FldbVfPBrXFRP4hNbCu3trL%2FDataset-Grower-Workflow%3Ftype%3Dwhiteboard%26node-id%3D0%253A1%26t%3DB6lvnCnQDdxCI1wl-1" allowfullscreen></iframe>
