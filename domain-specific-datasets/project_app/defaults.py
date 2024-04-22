import json

SEED_DATA_PATH = "seed_data.json"
PIPELINE_PATH = "pipeline.yaml"
REMOTE_CODE_PATHS = ["defaults.py", "domain.py", "pipeline.py"]

N_PERSPECTIVES = 5
N_TOPICS = 5
N_EXAMPLES = 5

with open(SEED_DATA_PATH) as f:
    DEFAULT_DATA = json.load(f)

with open("project_config.json") as f:
    PROJECT_CONFIG = json.load(f)

DEFAULT_DOMAIN = DEFAULT_DATA["domain"]
DEFAULT_PERSPECTIVES = DEFAULT_DATA["perspectives"]
DEFAULT_TOPICS = DEFAULT_DATA["topics"]
DEFAULT_EXAMPLES = DEFAULT_DATA["examples"]
DEFAULT_SYSTEM_PROMPT = DEFAULT_DATA["domain_expert_prompt"]
