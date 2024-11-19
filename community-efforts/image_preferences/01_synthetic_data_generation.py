import base64
import hashlib
import os
import random
from io import BytesIO
from typing import Any, Dict, List, Optional

from distilabel.llms import InferenceEndpointsLLM
from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, KeepColumns, LoadDataFromHub, StepInput, step
from distilabel.steps.base import StepInput
from distilabel.steps.tasks import Task, TextGeneration
from distilabel.steps.typing import StepOutput
from PIL import Image
from pydantic import validate_call

## At the time of writing this, the distilabel library does not support the image generation endpoint.
## This is a temporary fix to allow us to use the image generation endpoint.


class InferenceEndpointsImageLLM(InferenceEndpointsLLM):
    @validate_call
    async def agenerate(
        self,
        input: Dict[str, Any],
        negative_prompt: Optional[str] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        num_inference_steps: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        prompt = input.get("prompt")
        image = await self._aclient.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return [{"image": img_str}]


class ImageGeneration(Task):
    @property
    def inputs(self) -> List[str]:
        return ["prompt"]

    @property
    def outputs(self) -> List[str]:
        return ["image", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> Dict[str, str]:
        return {"prompt": input["prompt"]}

    def format_output(
        self, output: Dict[str, Any], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        image_str = output.get("image")
        image = None
        if image_str:
            image_bytes = base64.b64decode(image_str)
            image = Image.open(BytesIO(image_bytes))
        return {"image": image, "model_name": self.llm.model_name}

    def process(self, *args: StepInput) -> "StepOutput":
        inputs = args[0] if args else []
        formatted_inputs = self._format_inputs(inputs)

        outputs = self.llm.generate_outputs(
            inputs=formatted_inputs,
            num_generations=self.num_generations,
            **self.llm.get_generation_kwargs(),
        )

        task_outputs = []
        for input, input_outputs in zip(inputs, outputs):
            formatted_outputs = self._format_outputs(input_outputs, input)
            for formatted_output in formatted_outputs:
                if "image" in formatted_output and formatted_output["image"]:
                    # use prompt as filename
                    prompt_hash = hashlib.md5(input["prompt"].encode()).hexdigest()
                    self.save_artifact(
                        name="images",
                        write_function=lambda path: formatted_output["image"].save(
                            path / f"{prompt_hash}.jpeg"
                        ),
                        metadata={"type": "image", "library": "diffusers"},
                    )
                    formatted_output["image"] = {
                        "path": f"artifacts/{self.name}/images/{prompt_hash}.jpeg"
                    }

                task_output = {
                    **input,
                    **formatted_output,
                    "model_name": self.llm.model_name,
                }
                task_outputs.append(task_output)
        yield task_outputs


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

model_id = "Qwen/Qwen2.5-72B-Instruct"  # "meta-llama/Meta-Llama-3.1-70B-Instruct"


llm = InferenceEndpointsLLM(
    model_id=model_id,
    tokenizer_id=model_id,
    generation_kwargs={"temperature": 0.8, "max_new_tokens": 2048},
)

## We will use the Flux Dev and the stable diffusion 3.5 large model for the image generation task

sd = InferenceEndpointsImageLLM(
    base_url="https://nuiqd5tp37m6fxn4.us-east-1.aws.endpoints.huggingface.cloud",
    api_key=os.getenv("HF_TOKEN"),
)

flux_dev = InferenceEndpointsImageLLM(
    base_url="https://f94i5ss7a040r0v5.us-east-1.aws.endpoints.huggingface.cloud",
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
1. Preserve the main subject and action of the original prompt.
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

## Let's create the pipeline to generate the quality and style prompts

with Pipeline(name="prompt-augmentation") as pipeline:
    load_data = LoadDataFromHub(
        repo_id="fal/imgsys-results", name="load_dataset", num_examples=2
    )

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

    quality_augmentation = TextGeneration(
        llm=llm,
        template=quality_prompt,
        columns=["style_prompt"],
        name="quality_augmentation",
        output_mappings={"generation": "quality_prompt"},
    )

    style_augmentation = TextGeneration(
        llm=llm,
        template=style_prompt,
        columns=["prompt", "category", "subcategory"],
        name="style_augmentation",
        output_mappings={"generation": "style_prompt"},
    )

    image_gen_quality_dev = ImageGeneration(
        llm=flux_dev,
        input_mappings={"prompt": "quality_prompt"},
        output_mappings={"image": "image_quality_dev"},
    )

    image_gen_style_dev = ImageGeneration(
        llm=flux_dev,
        input_mappings={"prompt": "style_prompt"},
        output_mappings={"image": "image_style_dev"},
    )

    image_gen_quality_sd = ImageGeneration(
        llm=sd,
        input_mappings={"prompt": "quality_prompt"},
        output_mappings={"image": "image_quality_sd"},
    )

    image_gen_style_sd = ImageGeneration(
        llm=sd,
        input_mappings={"prompt": "style_prompt"},
        output_mappings={"image": "image_style_sd"},
    )

    group_columns = GroupColumns(columns=["model_name"])
    keep_columns = KeepColumns(
        columns=["prompt", "category", "subcategory", "style_prompt", "quality_prompt"]
    )
    group_columns_2 = GroupColumns(columns=["model_name"])

    (
        load_data
        >> category_selector
        >> style_augmentation
        >> quality_augmentation
        >> group_columns
        >> keep_columns
        >> [
            image_gen_quality_dev,
            image_gen_style_dev,
            image_gen_quality_sd,
            image_gen_style_sd,
        ]
        >> group_columns_2
    )

## Let's run the pipeline and push the resulting dataset to the hub

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=True)
    dataset_name = "davidberenstein1957/img_prefs_style"
    distiset.push_to_hub(dataset_name, include_script=True, generate_card=False)
