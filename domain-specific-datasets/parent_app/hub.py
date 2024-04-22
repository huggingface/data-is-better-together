from huggingface_hub import duplicate_space, HfApi


hf_api = HfApi()


def setup_dataset_on_hub(repo_id, hub_token):
    # create an empty dataset repo on the hub
    hf_api.create_repo(
        repo_id=repo_id,
        token=hub_token,
        repo_type="dataset",
    )


def duplicate_space_on_hub(source_repo, target_repo, hub_token, private=False):
    duplicate_space(
        from_id=source_repo, to_id=target_repo, token=hub_token, private=private
    )
