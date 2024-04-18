import streamlit as st


from defaults import (
    DEFAULT_DOMAIN,
)
from hub import setup_dataset_on_hub

################################################################################
# HEADER
################################################################################

st.header("🧑‍🌾 Domain Data Grower")
st.divider()
st.subheader(
    "Step 1. Create Dataset repo on the hub for your domain specific dataset",
)
st.write(
    "Define the project details, including the project name, domain, and API credentials"
)

################################################################################
# CONFIGURATION
################################################################################

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
        f"Dataset seed created and pushed to the Hub. Check it out [here](https://huggingface.co/{hub_username}/{project_name}).  Hold on the repo_id: {repo_id}, we will need it in the next steps."
    )

    st.session_state["created_dataset"] = True
    st.session_state["repo_id"] = repo_id
    st.session_state["hub_username"] = hub_username
    st.session_state["hub_token"] = hub_token
    st.session_state["project_name"] = project_name

    st.divider()

    st.write(
        "Now that you have created the dataset repo, you can start defining the domain expertise, perspectives, topics, and examples:"
    )
    st.page_link(
        page="pages/2_👩🏼‍🔬 Describe Domain.py",
    )
