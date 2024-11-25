import base64
import hashlib
import os
import random
from io import BytesIO
from typing import Any, Dict, List, Optional

from distilabel.llms import InferenceEndpointsLLM

# from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromHub, StepInput
from distilabel.steps.base import StepInput
from distilabel.steps.tasks import Task
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
            height=int(height) if height else None,
            width=int(width) if width else None,
            num_inference_steps=int(num_inference_steps)
            if num_inference_steps
            else None,
            guidance_scale=float(guidance_scale) if guidance_scale else None,
            seed=random.randint(0, 1000000),
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

sd = InferenceEndpointsImageLLM(
    base_url="https://el8g78juu06xfxtx.us-east-1.aws.endpoints.huggingface.cloud",
    api_key=os.getenv("HF_TOKEN"),
)

flux_dev = InferenceEndpointsImageLLM(
    base_url="https://f94i5ss7a040r0v5.us-east-1.aws.endpoints.huggingface.cloud",
    api_key=os.getenv("HF_TOKEN"),
)

## Let's create the pipeline to generate the quality and style prompts

with Pipeline(name="image_preferences_synthetic_data_generation") as pipeline:
    load_data = LoadDataFromHub(name="load_dataset")

    image_gen_quality_dev = ImageGeneration(
        name="image_gen_quality_dev",
        llm=flux_dev,
        input_mappings={"prompt": "quality_prompt"},
        output_mappings={"image": "image_quality_dev"},
    )

    image_gen_simplified_dev = ImageGeneration(
        name="image_gen_simplified_dev",
        llm=flux_dev,
        input_mappings={"prompt": "simplified_prompt"},
        output_mappings={"image": "image_simplified_dev"},
    )

    image_gen_quality_sd = ImageGeneration(
        name="image_gen_quality_sd",
        llm=sd,
        input_mappings={"prompt": "quality_prompt"},
        output_mappings={"image": "image_quality_sd"},
    )

    image_gen_simplified_sd = ImageGeneration(
        name="image_gen_simplified_sd",
        llm=sd,
        input_mappings={"prompt": "simplified_prompt"},
        output_mappings={"image": "image_simplified_sd"},
    )

    group_columns_2 = GroupColumns(columns=["model_name"])

    (
        load_data
        >> [
            image_gen_quality_dev,
            image_gen_simplified_dev,
            image_gen_quality_sd,
            image_gen_simplified_sd,
        ]
        >> group_columns_2
    )

## Let's run the pipeline and push the resulting dataset to the hub

if __name__ == "__main__":
    num_examples = 15000
    batch_size = 5
    num_inference_steps = 25
    width = 1024
    height = 1024
    distiset = pipeline.run(
        use_cache=True,
        parameters={
            load_data.name: {
                "num_examples": num_examples,
                "repo_id": "data-is-better-together/imgsys-results-prompts-style_v2_part1_cleaned",
            },
            image_gen_quality_sd.name: {
                "llm": {
                    "generation_kwargs": {
                        "width": width,
                        "height": height,
                        "guidance_scale": 4.5,
                        "num_inference_steps": num_inference_steps,
                    },
                },
                "input_batch_size": batch_size,
            },
            image_gen_quality_dev.name: {
                "llm": {
                    "generation_kwargs": {
                        "guidance_scale": 4.5,
                        "num_inference_steps": num_inference_steps,
                    },
                },
                "input_batch_size": batch_size,
            },
            image_gen_simplified_sd.name: {
                "llm": {
                    "generation_kwargs": {
                        "width": width,
                        "height": height,
                        "guidance_scale": 3.5,
                        "num_inference_steps": num_inference_steps,
                    },
                },
                "input_batch_size": batch_size,
            },
            image_gen_simplified_dev.name: {
                "llm": {
                    "generation_kwargs": {
                        "guidance_scale": 3.5,
                        "num_inference_steps": num_inference_steps,
                    },
                },
                "input_batch_size": batch_size,
            },
        },
    )
    from pathlib import Path

    from PIL import Image

    dataset = distiset["default"]["train"]
    artifacts_path = Path(distiset.artifacts_path)

    def load_images(batch):
        for column in [
            "image_quality_dev",
            "image_simplified_dev",
            "image_quality_sd",
            "image_simplified_sd",
        ]:
            batch[f"{column}_loaded"] = []
            for i in range(len(batch[column])):
                if batch[column][i]:
                    batch[f"{column}_loaded"].append(
                        Image.open(
                            artifacts_path
                            / batch[column][i]["path"].replace("artifacts/", "")
                        )
                    )
                else:
                    batch[f"{column}_loaded"].append(None)
        return batch

    dataset = dataset.map(load_images, batched=True)
    dataset = dataset.remove_columns(
        [
            "image_quality_dev",
            "image_simplified_dev",
            "image_quality_sd",
            "image_simplified_sd",
        ]
    )
    dataset = dataset.rename_columns(
        {
            "image_quality_dev_loaded": "image_quality_dev",
            "image_simplified_dev_loaded": "image_simplified_dev",
            "image_quality_sd_loaded": "image_quality_sd",
            "image_simplified_sd_loaded": "image_simplified_sd",
        }
    )
    distiset["default"]["train"] = dataset
    distiset.artifacts_path = ""
    distiset.push_to_hub(
        repo_id="data-is-better-together/image-preferences",
        include_script=True,
        generate_card=False,
        token=os.getenv("HF_TOKEN"),
    )
