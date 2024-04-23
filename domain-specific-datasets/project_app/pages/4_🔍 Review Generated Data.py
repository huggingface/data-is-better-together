import streamlit as st

from defaults import PROJECT_NAME, ARGILLA_URL, DATASET_REPO_ID
from utils import project_sidebar
from hub import push_argilla_dataset_to_hub

st.set_page_config(
    page_title="Domain Data Grower",
    page_icon="🧑‍🌾",
)

project_sidebar()

################################################################################
# HEADER
################################################################################

st.header("🧑‍🌾 Domain Data Grower")
st.divider()

st.write(
    """Once you have reviewed the synthetic data in Argilla, you can publish the 
    generated dataset to the Hub."""
)


################################################################################
# Configuration
################################################################################

st.divider()
st.write("🔬 Argilla API details to push the generated dataset")
argilla_url = st.text_input("Argilla API URL", ARGILLA_URL)
argilla_api_key = st.text_input("Argilla API Key", "owner.apikey")
argilla_dataset_name = st.text_input("Argilla Dataset Name", PROJECT_NAME)
dataset_repo_id = st.text_input("Dataset Repo ID", DATASET_REPO_ID)
st.divider()

if st.button("🚀 Publish the generated dataset"):
    with st.spinner("Publishing the generated dataset..."):
        push_argilla_dataset_to_hub(
            name=argilla_dataset_name,
            repo_id=dataset_repo_id,
            url=argilla_url,
            api_key=argilla_api_key,
            workspace="admin",
        )
    st.success("The generated dataset has been published to the Hub.")
