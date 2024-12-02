from datasets import Dataset, load_dataset
from fast_langdetect import detect

dataset = load_dataset("fal/imgsys-results", split="train")
dataset = dataset.shuffle()
df = dataset.to_pandas()
df = df.drop_duplicates(subset=["prompt"])
df = df.reset_index(drop=True)
df = df[["prompt"]]
df = df.dropna(subset=["prompt"])
df["language"], df["score"] = zip(
    *df["prompt"].apply(lambda x: detect(x.replace("\n", "")).values())
)
df = df[df["language"] == "en"]
df = df["prompt"]
dataset = Dataset.from_pandas(df)
dataset.push_to_hub(
    "data-is-better-together/imgsys-results-prompts-shuffled-cleaned-deduplicated-english"
)
