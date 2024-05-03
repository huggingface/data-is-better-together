import streamlit as st
import requests


readme_location = "https://raw.githubusercontent.com/huggingface/data-is-better-together/51f29e67165d8277d9f9d1e4be60869f4b705a08/domain-specific-datasets/README.md"


def open_markdown_file(url):
    response = requests.get(url)
    return response.text


readme = open_markdown_file(readme_location)

st.markdown(readme)
