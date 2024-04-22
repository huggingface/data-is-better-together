from defaults import (
    DEFAULT_DOMAIN,
)
from hub import (
    setup_dataset_on_hub,
    duplicate_space_on_hub,
    add_project_config_to_dataset_repo,
)

import streamlit as st

st.set_page_config("Domain Data Grower", page_icon="🧑‍🌾")
st.header("🧑‍🌾 Domain Data Grower")
st.divider()

################################################################################
# APP MARKDOWN
################################################################################

st.header("🌱 Create a domain specific dataset")

st.markdown(
    """This space will set up your domain specific dataset project. It will 
create the resources that you need to build a dataset. Those resources include: 
    
- A dataset repository on the Hub
- Another space to define expert domain and run generation pipelines    

For a complete overview of the project. Check out the README 
"""
)

st.page_link(
    "pages/🧑‍🌾 Domain Data Grower.py",
    label="🧑‍🌾 Domain Data Grower",
    icon="🧑‍🌾",
)

################################################################################
# CONFIGURATION
################################################################################

st.subheader("🌾 Project Configuration")

project_name = st.text_input("Project Name", DEFAULT_DOMAIN)
hub_username = st.text_input("Hub Username", "argilla")
hub_token = st.text_input("Hub Token", type="password")
private_selector = st.checkbox("Private Space", value=False)

if st.button("🤗 Setup Project Resources"):
    repo_id = f"{hub_username}/{project_name}"

    setup_dataset_on_hub(
        repo_id=repo_id,
        hub_token=hub_token,
    )

    st.success(
        f"Dataset seed created and pushed to the Hub. Check it out [here](https://huggingface.co/datasets/{hub_username}/{project_name}).  Hold on the repo_id: {repo_id}, we will need it in the next steps."
    )

    space_name = f"{project_name}_config_space"

    duplicate_space_on_hub(
        source_repo="argilla/domain-specific-template",
        target_repo=space_name,
        hub_token=hub_token,
        private=private_selector,
    )

    st.success(
        f"Configuration Space created. Check it out [here](https://huggingface.co/spaces/{hub_username}/{space_name})."
    )

    argilla_name = f"{project_name}_argilla_space"

    duplicate_space_on_hub(
        source_repo="argilla/argilla-template-space",
        target_repo=argilla_name,
        hub_token=hub_token,
        private=private_selector,
    )

    st.success(
        f"Argilla Space created. Check it out [here](https://huggingface.co/spaces/{hub_username}/{argilla_name})."
    )

    add_project_config_to_dataset_repo(
        repo_id=repo_id,
        hub_token=hub_token,
        project_name=project_name,
        argilla_space_repo_id=argilla_name,
        project_space_repo_id=space_name,
    )
