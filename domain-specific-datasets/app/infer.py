import requests

API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
)
headers = {"Authorization": "Bearer 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"}


def query(question):
    payload = {
        "inputs": question,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["generated_text"]
