from datasets import Dataset, load_dataset
from fast_langdetect import detect
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])

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
embeddings = model.encode(df["prompt"].tolist(), show_progress_bar=True)
df["embedding"] = embeddings.tolist()
dataset = Dataset.from_pandas(df)
dataset.push_to_hub(
    "data-is-better-together/imgsys-results-prompts-shuffled-cleaned-deduplicated-english"
)
