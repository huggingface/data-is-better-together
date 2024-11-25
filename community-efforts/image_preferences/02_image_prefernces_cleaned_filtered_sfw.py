from collections import defaultdict

from datasets import load_dataset
from transformers import pipeline

pipe_text = pipeline(
    "text-classification",
    model="ezb/NSFW-Prompt-Detector",
    device="mps",
)
pipe_text_2 = pipeline(
    "text-classification",
    model="michellejieli/NSFW_text_classifier",
    device="mps",
)
pipe_image = pipeline(
    "image-classification",
    model="MichalMlodawski/nsfw-image-detection-large",
    device="mps",
)

label_to_category_text = {
    "LABEL_0": "Safe",
    "LABEL_1": "Questionable",
    "LABEL_2": "Unsafe",
}


def clean_dataset(batch):
    try:
        batch["nsfw_text"] = []
        batch["nsfw_image"] = []
        evaluated_results_image = defaultdict(list)
        evaluated_results_text = defaultdict(list)

        image_columns = [
            "image_quality_dev",
            "image_simplified_dev",
            "image_quality_sd",
            "image_simplified_sd",
        ]

        for image_column in image_columns:
            results_image = pipe_image(batch[image_column])
            evaluated_results_image[image_column] = [
                res[0]["label"] in ["UNSAFE", "QUESTIONABLE"] for res in results_image
            ]

        try:
            results_text = pipe_text(batch["prompt"])
            results_text_2 = pipe_text_2(batch["prompt"])
            evaluated_results_text["text"] = [
                res["label"] == "NSFW" for res in results_text
            ]
            evaluated_results_text["text_2"] = [
                res["label"] == "NSFW" for res in results_text_2
            ]
        except Exception:
            try:
                results_text_2 = pipe_text_2(batch["prompt"])
                evaluated_results_text["text_2"] = [
                    res["label"] == "NSFW" for res in results_text_2
                ]
                evaluated_results_text["text"] = [False] * len(results_text_2)
            except Exception:
                try:
                    results_text = pipe_text(batch["prompt"])
                    evaluated_results_text["text"] = [
                        res["label"] == "NSFW" for res in results_text
                    ]
                    evaluated_results_text["text_2"] = [False] * len(results_text)
                except Exception:
                    for item in batch["prompt"]:
                        try:
                            evaluated_results_text["text"].append(
                                pipe_text(item)["label"] == "NSFW"
                            )
                        except Exception:
                            evaluated_results_text["text"].append(True)
                        try:
                            evaluated_results_text["text_2"].append(
                                pipe_text_2(item)["label"] == "NSFW"
                            )
                        except Exception:
                            evaluated_results_text["text_2"].append(True)

        for i in range(len(evaluated_results_text["text"])):
            if any(evaluated_results_text[col][i] for col in evaluated_results_text):
                batch["nsfw_text"].append(True)
            else:
                batch["nsfw_text"].append(False)
        for i in range(len(evaluated_results_image["image_quality_dev"])):
            if any(evaluated_results_image[col][i] for col in evaluated_results_image):
                batch["nsfw_image"].append(True)
            else:
                batch["nsfw_image"].append(False)
    except Exception as e:
        raise Exception(e)
    return batch


ds = load_dataset("data-is-better-together/image-preferences", split="train")
df = ds.filter(
    lambda x: x["image_quality_dev"]
    and x["image_simplified_dev"]
    and x["image_quality_sd"]
    and x["image_simplified_sd"]
)
ds = df.map(clean_dataset, batched=True, batch_size=100)
# ds = ds.filter(lambda x: not x["nsfw"])
# ds = ds.remove_columns(["nsfw"])
ds.push_to_hub(
    "data-is-better-together/image-preferences-unfiltered",
    split="cleaned",
    private=True,
)
