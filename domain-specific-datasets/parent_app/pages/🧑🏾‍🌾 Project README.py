import streamlit as st
import requests


readme_location = "https://raw.githubusercontent.com/huggingface/data-is-better-together/4d7848149dcfe575b86517ca15e4aaa09dc9db74/domain-specific-datasets/README.md"


def open_markdown_file(url):
    response = requests.get(url)
    return response.text


readme = open_markdown_file(readme_location)

st.markdown(readme)
