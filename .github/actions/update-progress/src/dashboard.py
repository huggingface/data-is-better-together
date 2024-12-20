import os
import argilla as rg
from huggingface_hub import HfApi, hf_hub_download
import httpx
import stamina
import polars as pl
from tqdm.contrib.concurrent import thread_map
from argilla._exceptions import ArgillaAPIError
from datetime import datetime, timezone

# Enable HF transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Validate environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

if ARGILLA_API_KEY := os.environ.get("ARGILLA_API_KEY"):
    client = rg.Argilla(
        api_url="https://data-is-better-together-fineweb-c.hf.space",
        api_key=ARGILLA_API_KEY,
        timeout=120,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
    )
else:
    raise ValueError("ARGILLA_API_KEY environment variable is not set")


def get_dataset_for_language(language_code):
    all_datasets = client.datasets.list()
    dataset = [
        dataset for dataset in all_datasets if dataset.name.startswith(language_code)
    ]
    if len(dataset) != 1:
        raise ValueError(
            f"Found {len(dataset)} datasets for language code {language_code}"
        )
    dataset_name = dataset[0].name
    return client.datasets(dataset_name)


# Get all datasets
all_datasets = client.datasets.list()
language_datasets_names = [dataset.name for dataset in all_datasets]


@stamina.retry(on=(httpx.HTTPStatusError, ArgillaAPIError), attempts=3, wait_initial=5)
def get_dataset_progress(language_dataset_name):
    dataset = client.datasets(language_dataset_name)
    return {
        "language_dataset_name": language_dataset_name,
        **dataset.progress(with_users_distribution=True),
    }


def flatten_user_stats(dataset):
    dataset_name = dataset["language_dataset_name"]
    current_timestamp = datetime.now(timezone.utc)
    user_stats = []

    if dataset["users"]:
        user_stats.extend(
            {
                "language_dataset_name": dataset_name,
                "username": str(username),
                "submitted": int(
                    stats["completed"]["submitted"] + stats["pending"]["submitted"]
                ),
                "total": int(dataset["total"]),
                "timestamp": current_timestamp,
            }
            for username, stats in dataset["users"].items()
        )
    else:
        user_stats.append(
            {
                "language_dataset_name": dataset_name,
                "username": None,
                "submitted": 0,
                "total": int(dataset["total"]),
                "timestamp": current_timestamp,
            }
        )

    return user_stats


def update_progress_data(new_data, filename="argilla_progress.ndjson"):
    # Process new data
    all_user_stats = []
    for dataset in new_data:
        all_user_stats.extend(flatten_user_stats(dataset))

    new_df = pl.DataFrame(
        all_user_stats,
        schema={
            "language_dataset_name": pl.Utf8,
            "username": pl.Utf8,
            "submitted": pl.Int64,
            "total": pl.Int64,
            "timestamp": pl.Datetime,
        },
    )

    try:
        fname = hf_hub_download(
            repo_id="davanstrien/progress",
            filename="argilla_progress.ndjson",
            repo_type="dataset",
        )
        existing_df = pl.read_ndjson(fname)
        combined_df = pl.concat([existing_df, new_df])
    except FileNotFoundError:
        print("No existing data found, creating new dataset")
        combined_df = new_df
    except Exception as e:
        print(f"Error loading existing data: {e}")
        combined_df = new_df

    combined_df.write_ndjson(filename)
    return combined_df


def main():
    print("Starting data collection...")
    all_data = thread_map(get_dataset_progress, language_datasets_names, max_workers=1)

    print("Updating progress data...")
    df = update_progress_data(all_data)
    df = df.sort("language_dataset_name")

    print("Saving data...")
    df.write_ndjson("argilla_progress.ndjson")

    print("Uploading to Hugging Face Hub...")
    api = HfApi()
    api.create_repo(
        "data-is-better-together/fineweb-c-progress", repo_type="dataset", exist_ok=True
    )
    api.upload_file(
        path_or_fileobj="argilla_progress.ndjson",
        repo_id="data-is-better-together/fineweb-c-progress",
        repo_type="dataset",
        path_in_repo="argilla_progress.ndjson",
    )
    print("Done!")


if __name__ == "__main__":
    main()
