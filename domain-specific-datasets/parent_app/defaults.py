import json

SEED_DATA_PATH = "seed_data.json"

with open(SEED_DATA_PATH) as f:
    DEFAULT_DATA = json.load(f)
DEFAULT_DOMAIN = DEFAULT_DATA["domain"]
