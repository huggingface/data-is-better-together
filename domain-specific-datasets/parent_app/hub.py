import json
from tempfile import mktemp


from huggingface_hub import duplicate_space, HfApi


hf_api = HfApi()


def setup_dataset_on_hub(repo_id, hub_token):
    # create an empty dataset repo on the hub
    hf_api.create_repo(
        repo_id=repo_id,
        token=hub_token,
        repo_type="dataset",
    )

    # upload the seed data
    hf_api.upload_file(
        path_or_fileobj="seed_data.json",
        path_in_repo="seed_data.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=hub_token,
    )


def duplicate_space_on_hub(source_repo, target_repo, hub_token, private=False):
    duplicate_space(
        from_id=source_repo,
        to_id=target_repo,
        token=hub_token,
        private=private,
        exist_ok=True,
    )


def add_project_config_to_space_repo(
    dataset_repo_id,
    hub_token,
    project_name,
    argilla_space_repo_id,
    project_space_repo_id,
):
    #  upload the seed data and readme to the hub

    with open("project_config.json", "w") as f:
        json.dump(
            {
                "project_name": project_name,
                "argilla_space_repo_id": argilla_space_repo_id,
                "project_space_repo_id": project_space_repo_id,
                "dataset_repo_id": dataset_repo_id,
            },
            f,
        )

    hf_api.upload_file(
        path_or_fileobj="project_config.json",
        path_in_repo="project_config.json",
        token=hub_token,
        repo_id=project_space_repo_id,
        repo_type="space",
    )


def pull_seed_data_from_repo(repo_id, hub_token):
    tempfile_path = mktemp()
    # pull the dataset repo from the hub
    hf_api.hf_hub_download(
        repo_id=repo_id, token=hub_token, repo_type="dataset", filename=tempfile_path
    )
    return json.load(open(tempfile_path))


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
        path_in_repo="pipeline.py",
        token=hub_token,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"pipeline.py uploaded to {repo_id}")