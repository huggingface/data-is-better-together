import os
import json

SEED_DATA_PATH = "seed_data.json"
PIPELINE_PATH = "pipeline.yaml"
REMOTE_CODE_PATHS = ["requirements.txt"]
DIBT_PARENT_APP_URL = "https://argilla-domain-specific-datasets-welcome.hf.space/"
N_PERSPECTIVES = 5
N_TOPICS = 5
N_EXAMPLES = 5
CODELESS_DISTILABEL = os.environ.get("CODELESS_DISTILABEL", True)

################################################
# DEFAULTS ON FARMING
################################################

with open(SEED_DATA_PATH) as f:
    DEFAULT_DATA = json.load(f)

DEFAULT_DOMAIN = DEFAULT_DATA["domain"]
DEFAULT_PERSPECTIVES = DEFAULT_DATA["perspectives"]
if DEFAULT_PERSPECTIVES is None or len(DEFAULT_PERSPECTIVES) == 0:
    DEFAULT_PERSPECTIVES = [""]
DEFAULT_TOPICS = DEFAULT_DATA["topics"]
if DEFAULT_TOPICS is None or len(DEFAULT_TOPICS) == 0:
    DEFAULT_TOPICS = [""]
DEFAULT_EXAMPLES = DEFAULT_DATA["examples"]
if DEFAULT_EXAMPLES is None or len(DEFAULT_EXAMPLES) == 0:
    DEFAULT_EXAMPLES = [{"question": "", "answer": ""}]
DEFAULT_SYSTEM_PROMPT = DEFAULT_DATA["domain_expert_prompt"]

################################################
# PROJECT CONFIG FROM PARENT APP
################################################

try:
    with open("project_config.json") as f:
        PROJECT_CONFIG = json.load(f)

    PROJECT_NAME = PROJECT_CONFIG["project_name"]
    ARGILLA_SPACE_REPO_ID = PROJECT_CONFIG["argilla_space_repo_id"]
    DATASET_REPO_ID = PROJECT_CONFIG["dataset_repo_id"]
    ARGILLA_SPACE_NAME = ARGILLA_SPACE_REPO_ID.replace("/", "-").replace("_", "-")
    ARGILLA_URL = f"https://{ARGILLA_SPACE_NAME}.hf.space"
    PROJECT_SPACE_REPO_ID = PROJECT_CONFIG["project_space_repo_id"]
    DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
    HUB_USERNAME = DATASET_REPO_ID.split("/")[0]
except FileNotFoundError:
    PROJECT_NAME = "DEFAULT_DOMAIN"
    ARGILLA_SPACE_REPO_ID = ""
    DATASET_REPO_ID = ""
    ARGILLA_URL = ""
    PROJECT_SPACE_REPO_ID = ""
    DATASET_URL = ""
    HUB_USERNAME = ""
