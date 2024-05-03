import os
import requests

API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
)


def query(question, hub_token: str):
    payload = {
        "inputs": question,
        "parameters": {
            "wait_for_model": True,
            "return_full_text": False,
        },
    }
    headers = {"Authorization": f"Bearer {hub_token}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "Error occurred while querying the model."
