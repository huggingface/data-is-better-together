import os
import requests

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
)
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def query(question):
    payload = {
        "inputs": question,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["generated_text"]
