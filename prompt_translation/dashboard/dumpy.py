import json
import logging
import os

import argilla as rg
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("*** Initializing Argilla session ***")
    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY"),
        extra_headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
    )

    logger.info("*** Fetching dataset from Argilla ***")
    dataset = rg.FeedbackDataset.from_argilla(
        os.getenv("SOURCE_DATASET"),
        workspace=os.getenv("SOURCE_WORKSPACE"),
    )
    logger.info("*** Filtering records by `response_status` ***")
    dataset = dataset.filter_by(response_status=["submitted"])  # type: ignore

    logger.info("*** Calculating users and annotation count ***")
    output = {}
    for record in dataset.records:
        for response in record.responses:
            if response.user_id not in output:
                output[response.user_id] = 0
            output[response.user_id] += 1

    for key in list(output.keys()):
        output[rg.User.from_id(key).username] = output.pop(key)

    logger.info("*** Users and annotation count successfully calculated! ***")

    logger.info("*** Dumping Python dict into `stats.json` ***")
    with open("stats.json", "w") as file:
        json.dump(output, file, indent=4)

    logger.info("*** Uploading `stats.json` to Hugging Face Hub ***")
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj="stats.json",
        path_in_repo="stats.json",
        repo_id="DIBT/prompt-collective-dashboard",
        repo_type="space",
    )
    logger.info("*** `stats.json` successfully uploaded to Hugging Face Hub! ***")