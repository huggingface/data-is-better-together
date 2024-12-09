# Open Image Preferences Dataset

## What is it?

This is a project for the community to contribute image preferences for an open source dataset, that could be used for training and evaluating text to image models. You can find a full blogpost [here](https://huggingface.co/blog/image-preferences).

## What did we achieve?

We achieved to annotate 10K preference pairs. You can take a look at the resulting dataset [here](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1-results), and [its version that is ready for training](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1-binarized). Additionally, we showcased the effectiveness along with a [FLUX-dev LoRA fine-tune](https://huggingface.co/data-is-better-together/open-image-preferences-v1-flux-dev-lora).

## How to use the dataset

The dataset is hosted on Hugging Face, and free for anyone to use under an Apache 2.0 license. Here are some [examples of how to use the dataset for fine-tuning or post-analysis](https://huggingface.co/blog/image-preferences#what-is-next).

## Which tools were used?

For the prompt ranking project, we used two tools to help us manage the annotation process.

- [Argilla](https://github.com/argilla-io/argilla): an open-source data annotation tool that we used for the prompt ranking. Argilla has the option of using Hugging Face for authentication, which makes it easier for the community to contribute.
- [distilabel](https://github.com/argilla-io/distilabel): a tool for creating and sythetic datasets. We used distilabel to evolve prompt and to create the image preferences dataset.
- [Hugging Face Spaces](https://huggingface.co/spaces): a platform for hosting machine learning applications and demos. We used Spaces to host the Argilla tool for prompt ranking.