import json
from tempfile import mktemp
from huggingface_hub import HfApi

from defaults import REMOTE_CODE_PATHS, SEED_DATA_PATH


hf_api = HfApi()

with open("DATASET_README_BASE.md") as f:
    DATASET_README_BASE = f.read()


def create_readme(domain_seed_data, project_name, domain):
    # create a readme for the project that shows the domain and project name
    readme = DATASET_README_BASE
    readme += f"# {project_name}\n\n## Domain: {domain}"
    perspectives = domain_seed_data.get("perspectives")
    topics = domain_seed_data.get("topics")
    examples = domain_seed_data.get("examples")
    if perspectives:
        readme += "\n\n## Perspectives\n\n"
        for p in perspectives:
            readme += f"- {p}\n"
    if topics:
        readme += "\n\n## Topics\n\n"
        for t in topics:
            readme += f"- {t}\n"
    if examples:
        readme += "\n\n## Examples\n\n"
        for example in examples:
            readme += f"### {example['question']}\n\n{example['answer']}\n\n"
    temp_file = mktemp()

    with open(temp_file, "w") as f:
        f.write(readme)
    return temp_file


def setup_dataset_on_hub(repo_id, hub_token):
    # create an empty dataset repo on the hub
    hf_api.create_repo(
        repo_id=repo_id,
        token=hub_token,
        repo_type="dataset",
        exist_ok=True,
    )


def push_dataset_to_hub(
    domain_seed_data_path,
    project_name,
    domain,
    pipeline_path,
    hub_username,
    hub_token: str,
):
    repo_id = f"{hub_username}/{project_name}"

    setup_dataset_on_hub(repo_id=repo_id, hub_token=hub_token)

    #  upload the seed data and readme to the hub
    hf_api.upload_file(
        path_or_fileobj=domain_seed_data_path,
        path_in_repo="seed_data.json",
        token=hub_token,
        repo_id=repo_id,
        repo_type="dataset",
    )

    # upload the readme to the hub
    domain_seed_data = json.load(open(domain_seed_data_path))
    hf_api.upload_file(
        path_or_fileobj=create_readme(
            domain_seed_data=domain_seed_data, project_name=project_name, domain=domain
        ),
        path_in_repo="README.md",
        token=hub_token,
        repo_id=repo_id,
        repo_type="dataset",
    )


def push_pipeline_to_hub(
    pipeline_path,
    hub_username,
    hub_token: str,
    project_name,
):
    repo_id = f"{hub_username}/{project_name}"

    # upload the pipeline to the hub
    hf_api.upload_file(
        path_or_fileobj=pipeline_path,
        path_in_repo="pipeline.yaml",
        token=hub_token,
        repo_id=repo_id,
        repo_type="dataset",
    )

    for code_path in REMOTE_CODE_PATHS:
        hf_api.upload_file(
            path_or_fileobj=code_path,
            path_in_repo=code_path,
            token=hub_token,
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"Dataset uploaded to {repo_id}")


def pull_seed_data_from_repo(repo_id, hub_token):
    # pull the dataset repo from the hub
    hf_api.hf_hub_download(
        repo_id=repo_id, token=hub_token, repo_type="dataset", filename=SEED_DATA_PATH
    )
    return json.load(open(SEED_DATA_PATH))
