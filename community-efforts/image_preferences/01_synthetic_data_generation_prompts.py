import os
import random

os.environ["DISTILABEL_LOG_LEVEL"] = "DEBUG"

from distilabel.llms import InferenceEndpointsLLM

# from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, KeepColumns, LoadDataFromHub, StepInput, step
from distilabel.steps.base import StepInput
from distilabel.steps.tasks import TextGeneration
from distilabel.steps.typing import StepOutput

## At the time of writing this, the distilabel library does not support the image generation endpoint.
## This is a temporary fix to allow us to use the image generation endpoint.

## Let's determine the categories and subcategories for the image generation task
# https://huggingface.co/spaces/google/sdxl/blob/main/app.py#L55
categories = {
    # included
    "Cinematic": [
        # included
        "emotional",
        "harmonious",
        "vignette",
        "highly detailed",
        "high budget",
        "bokeh",
        "cinemascope",
        "moody",
        "epic",
        "gorgeous",
        "film grain",
        "grainy",
    ],
    # included
    "Photographic": [
        # included
        "film",
        "bokeh",
        "professional",
        "4k",
        "highly detailed",
        ## not included
        "Landscape",
        "Portrait",
        "Macro",
        "Portra",
        "Gold",
        "ColorPlus",
        "Ektar",
        "Superia",
        "C200",
        "CineStill",
        "CineStill 50D",
        "CineStill 800T",
        "Tri-X",
        "HP5",
        "Delta",
        "T-Max",
        "Fomapan",
        "StreetPan",
        "Provia",
        "Ektachrome",
        "Velvia",
    ],
    # included
    "Anime": [
        # included
        "anime style",
        "key visual",
        "vibrant",
        "studio anime",
        "highly detailed",
    ],
    # included
    "Manga": [
        # included
        "vibrant",
        "high-energy",
        "detailed",
        "iconic",
        "Japanese comic style",
    ],
    # included
    "Digital art": [
        # included
        "digital artwork",
        "illustrative",
        "painterly",
        "matte painting",
        "highly detailed",
    ],
    # included
    "Pixel art": [
        # included
        "low-res",
        "blocky",
        "pixel art style",
        "8-bit graphics",
    ],
    # included
    "Fantasy art": [
        # included
        "magnificent",
        "celestial",
        "ethereal",
        "painterly",
        "epic",
        "majestic",
        "magical",
        "fantasy art",
        "cover art",
        "dreamy",
    ],
    # included
    "Neonpunk": [
        # included
        "cyberpunk",
        "vaporwave",
        "neon",
        "vibes",
        "vibrant",
        "stunningly beautiful",
        "crisp",
        "detailed",
        "sleek",
        "ultramodern",
        "magenta highlights",
        "dark purple shadows",
        "high contrast",
        "cinematic",
        "ultra detailed",
        "intricate",
        "professional",
    ],
    # included
    "3D Model": [
        # included
        "octane render",
        "highly detailed",
        "volumetric",
        "dramatic lighting",
    ],
    # not included
    "Painting": [
        "Oil",
        "Acrylic",
        "Watercolor",
        "Digital",
        "Mural",
        "Sketch",
        "Gouache",
        "Renaissance",
        "Baroque",
        "Romanticism",
        "Impressionism",
        "Expressionism",
        "Cubism",
        "Surrealism",
        "Pop Art",
        "Minimalism",
        "Realism",
        "Encaustic",
        "Tempera",
        "Fresco",
        "Ink Wash",
        "Spray Paint",
        "Mixed Media",
    ],
    # not included
    "Animation": [
        # not included
        "Animation",
        "Stop motion",
        "Claymation",
        "Pixel Art",
        "Vector",
        "Hand-drawn",
        "Cutout",
        "Whiteboard",
    ],
    # not included
    "Illustration": [
        # not included
        "Book",
        "Comics",
        "Editorial",
        "Advertising",
        "Technical",
        "Fantasy",
        "Scientific",
        "Fashion",
        "Storyboard",
        "Concept Art",
        "Manga",
        "Anime",
        "Digital",
        "Vector",
        "Design",
    ],
}

## We will use the Qwen2.5-72B-Instruct model for the text generation task, this will help us to generate the quality and style prompts

model_id = (
    "meta-llama/Llama-3.1-8B-Instruct"
)  # "meta-llama/Meta-Llama-3.1-70B-Instruct"


llm = InferenceEndpointsLLM(
    # model_id=model_id,
    # tokenizer_id=model_id,
    generation_kwargs={"temperature": 0.8, "max_new_tokens": 2048},
    base_url="https://rti2mzernqmo00qy.us-east-1.aws.endpoints.huggingface.cloud",
    api_key=os.getenv("HF_TOKEN"),
)


## We will use two types of prompts: quality and style. The quality prompt will help us to generate the quality-enhanced prompts and the style prompt will help us to generate the style-enhanced prompts.
quality_prompt = """
You are an expert at refining prompts for image generation models. Your task is to enhance the given prompt by adding descriptive details and quality-improving elements, while maintaining the original intent and core concept.

Follow these guidelines:
1. Preserve the main subject and action of the original prompt.
2. Add specific, vivid details to enhance visual clarity.
3. Incorporate elements that improve overall image quality and aesthetics.
4. Keep the prompt concise and avoid unnecessary words.
5. Use modifiers that are appropriate for the subject matter.

Example modifiers (use as reference, adapt based on some aspect that's suitable for the original prompt):
- Lighting: "soft golden hour light", "dramatic chiaroscuro", "ethereal glow"
- Composition: "rule of thirds", "dynamic perspective", "symmetrical balance"
- Texture: "intricate details", "smooth gradients", "rich textures"
- Color: "vibrant color palette", "monochromatic scheme", "complementary colors"
- Atmosphere: "misty ambiance", "serene mood", "energetic atmosphere"
- Technical: "high resolution", "photorealistic", "sharp focus"

The enhanced prompt should be short, concise, direct, avoid unnecessary words and written as it was a human expert writing the prompt.

Output only one enhanced prompt without any additional text or explanations.

## Original Prompt
{{ style_prompt }}

## Quality-Enhanced Prompt
"""

style_prompt = """
You are an expert at refining prompts for image generation models. Your task is to enhance the given prompt by transforming it into a specific artistic style, technique, or genre, while maintaining the original core concept.

Follow these guidelines:
1. Preserve the main subject and action of the original prompt but rewrite stylistic elements already present in the prompt.
2. Transform the prompt into a distinctive visual style (e.g., impressionism, surrealism, cyberpunk, art nouveau).
3. Incorporate style-specific elements and techniques.
4. Keep the prompt concise and avoid unnecessary words.
5. Use modifiers that are appropriate for the chosen style.

You should use the following style, technique, genre to enhance the prompt:
{{ category }} / {{ subcategory }}

The enhanced prompt should be short, concise, direct, avoid unnecessary words and written as it was a human expert writing the prompt.

Output only one style-enhanced prompt without any additional text or explanations.

## Original Prompt
{{ prompt }}

## Style-Enhanced Prompt
"""

simplification_prompt = """
You are an expert at simplifying image descriptions. Your task is to simplify the description by removing any unnecessary words and phrases, while maintaining the original intent and core concept of the description.

Follow these guidelines:
1. Preserve the main subject of the original description.
2. Remove all any unnecessary words and phrases.
3. Ensure the simplified description could have been quickly written by a human.

## Original Description
{{ style_prompt }}

## Simplified Description
"""

## Let's create the pipeline to generate the quality and style prompts

with Pipeline(name="image_preferences_synthetic_data_generation") as pipeline:
    load_data = LoadDataFromHub(name="load_dataset")

    @step(inputs=["prompt"], outputs=["category", "subcategory", "prompt"])
    def CategorySelector(inputs: StepInput) -> "StepOutput":
        result = []
        for input in inputs:
            # Randomly select a category
            category = random.choice(list(categories.keys()))
            # Randomly select a subcategory from the chosen category
            subcategory = random.choice(categories[category])

            result.append(
                {
                    "category": category,
                    "subcategory": subcategory,
                    "prompt": input["prompt"],
                }
            )
        yield result

    category_selector = CategorySelector(name="category_selector")

    style_augmentation = TextGeneration(
        llm=llm,
        template=style_prompt,
        columns=["prompt", "category", "subcategory"],
        name="style_augmentation",
        output_mappings={"generation": "style_prompt"},
        input_batch_size=4,
    )

    simplification_augmentation = TextGeneration(
        llm=llm,
        template=simplification_prompt,
        columns=["style_prompt"],
        name="simplification_augmentation",
        output_mappings={"generation": "simplified_prompt"},
        input_batch_size=2,
    )

    quality_augmentation = TextGeneration(
        llm=llm,
        template=quality_prompt,
        columns=["style_prompt"],
        name="quality_augmentation",
        output_mappings={"generation": "quality_prompt"},
        input_batch_size=2,
    )

    group_columns = GroupColumns(columns=["model_name"])
    keep_columns = KeepColumns(
        columns=[
            "prompt",
            "category",
            "subcategory",
            "style_prompt",
            "quality_prompt",
            "simplified_prompt",
        ]
    )

    (
        load_data
        >> category_selector
        >> style_augmentation
        >> [quality_augmentation, simplification_augmentation]
        >> group_columns
        >> keep_columns
    )

## Let's run the pipeline and push the resulting dataset to the hub

if __name__ == "__main__":
    num_examples = 15000
    distiset = pipeline.run(
        use_cache=True,
        parameters={
            load_data.name: {
                "num_examples": num_examples,
                "repo_id": "data-is-better-together/imgsys-results-prompts-shuffled-cleaned-deduplicated-english",
            }
        },
    )
    dataset_name = "data-is-better-together/imgsys-results-prompts-style_v2_part1"
    distiset.push_to_hub(
        repo_id=dataset_name,
        include_script=True,
        generate_card=False,
        token=os.getenv("HF_TOKEN"),
    )
