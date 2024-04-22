from defaults import (
    DEFAULT_DOMAIN,
)
from hub import setup_dataset_on_hub, duplicate_space_on_hub

import streamlit as st

st.set_page_config("Domain Data Grower", page_icon="🧑‍🌾")

################################################################################
# APP MARKDOWN
################################################################################


introduction = """
## 🌱 Create a dataset seed for aligning models to a specific domain

This app helps you create a dataset seed for building diverse domain-specific datasets for aligning models.
Alignment datasets are used to fine-tune models to a specific domain or task, but as yet, there's a shortage of diverse datasets for this purpose.

## 🚜 How it works

You can create a dataset seed by defining the domain expertise, perspectives, topics, and examples for your domain-specific dataset. 
The dataset seed is then used to generate synthetic data for training a language model.

## 🗺️ The process

Define the project details, including the project name, domain, and API credentials. A dataset repository will be created, and also a personalised Streamlit app in your Hugging Face profile, with which you'll be able to define the domain expertise, perspectives, topics, and examples for your domain-specific dataset, and generate the synthetic data.

## 👩🏽‍🌾 Current Projects
WIP

"""

instructions_project_details = """
Define the project details, including the project name, domain, and API credentials
"""

step1_subheader = "### Step 1: Create Dataset repo on the hub for your domain specific dataset"

step2_subheader = "### Step 2: Duplicate the Streamlit app for your domain specific dataset"

instructions_duplication = """
Define the project details, including the project name, domain, and API credentials
"""

## 🌱 Create a dataset seed for aligning models to a specific domain

################################################################################
# HEADER
################################################################################


st.markdown("# 🧑‍🌾 Domain Data Grower")
st.markdown(introduction)


################################################################################
# CONFIGURATION
################################################################################

st.header("🌾 Create the Dataset and the Configuration Space")
st.markdown(step1_subheader)
st.markdown(instructions_project_details)

project_name = st.text_input("Project Name", DEFAULT_DOMAIN)
hub_username = st.text_input("Hub Username", "argilla")
hub_token = st.text_input("Hub Token", type="password")

if st.button("🤗 Create Dataset Repo"):
    repo_id = f"{hub_username}/{project_name}"

    setup_dataset_on_hub(
        repo_id=repo_id,
        hub_token=hub_token,
    )

    st.success(
        f"Dataset seed created and pushed to the Hub. Check it out [here](https://huggingface.co/datasets/{hub_username}/{project_name}).  Hold on the repo_id: {repo_id}, we will need it in the next steps."
    )

st.markdown(step2_subheader)
st.markdown(instructions_duplication)

space_name = st.text_input("HF Space Name", DEFAULT_DOMAIN)
private_selector = st.checkbox("Private Space", value=False)

if st.button("🤗 Create Configuration Space"):
    repo_id = f"{hub_username}/{project_name}"

    duplicate_space_on_hub(
        source_repo="BEN",
        target_repo=project_name,
        hub_token=hub_token,
        private=private_selector
    )

    st.success(
        f"Configuration Space created. Check it out [here](https://huggingface.co/datasets/{repo_id})."
    )

