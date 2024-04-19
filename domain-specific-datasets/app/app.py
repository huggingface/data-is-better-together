import streamlit as st

st.set_page_config("Domain Data Grower", page_icon="🧑‍🌾")

st.header("🧑‍🌾 Domain Data Grower")
st.divider()

st.markdown(
    """
## 🌱 Create a dataset seed for aligning models to a specific domain

This app helps you create a dataset seed for building diverse domain-specific datasets for aligning models.
Alignment datasets are used to fine-tune models to a specific domain or task, but as yet, there's a shortage of diverse datasets for this purpose.
"""
)
st.markdown(
    """
## 🚜 How it works

You can create a dataset seed by defining the domain expertise, perspectives, topics, and examples for your domain-specific dataset. 
The dataset seed is then used to generate synthetic data for training a language model.

"""
)
st.markdown(
    """
## 🗺️ The process

### Step 1: Setup the project

Define the project details, including the project name, domain, and API credentials. Create Dataset Repo on the Hub.
"""
)
# TODO: add a link to the DIBT parent app

st.markdown(
    """
### Step 2: Describe the Domain

Define the domain expertise, perspectives, topics, and examples for your domain-specific dataset. 
You can collaborate with domain experts to define the domain expertise and perspectives.
"""
)

st.page_link(
    "pages/2_👩🏼‍🔬 Describe Domain.py",
    label="👩🏼‍🔬 Describe Domain",
    icon="👩🏼‍🔬",
)

st.markdown(
    """
### Step 3: Generate Synthetic Data

Use distilabel to generate synthetic data for your domain-specific dataset. 
You can run the pipeline locally or in this space to generate synthetic data.
"""
)

st.page_link(
    "pages/3_🌱 Generate Dataset.py",
    label="🌱 Generate Dataset",
    icon="🌱",
)

st.markdown(
    """
### Step 4: Review the Dataset

Use Argilla to review the generated synthetic data and provide feedback on the quality of the data.


"""
)

# TODO: add a link to argilla space based on the app configration
