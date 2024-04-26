import streamlit as st

from defaults import (
    ARGILLA_SPACE_REPO_ID,
    PROJECT_NAME,
    ARGILLA_URL,
    DIBT_PARENT_APP_URL,
    DATASET_URL,
    DATASET_REPO_ID,
    ARGILLA_SPACE_REPO_ID,
)


def project_sidebar():
    if PROJECT_NAME == "DEFAULT_DOMAIN":
        st.warning(
            "Please set up the project configuration in the parent app before proceeding."
        )
        st.stop()

    st.sidebar.subheader(f"A Data Growing Project in the domain of {PROJECT_NAME}")
    st.sidebar.markdown(
        """        
        This space helps you create a dataset seed for building diverse domain-specific datasets for aligning models.
        """
    )
    st.sidebar.link_button(f"📚 Dataset Repo", DATASET_URL)
    st.sidebar.link_button(f"🤖 Argilla Space", ARGILLA_URL)
    hub_username = DATASET_REPO_ID.split("/")[0]
    project_name = DATASET_REPO_ID.split("/")[1]
    st.session_state["project_name"] = project_name
    st.session_state["hub_username"] = hub_username
    st.session_state["hub_token"] = st.sidebar.text_input(
        "Hub Token", type="password", value=None
    )
    st.sidebar.link_button(
        "🤗 Get your Hub Token", "https://huggingface.co/settings/tokens"
    )
    if all(
        (
            st.session_state.get("project_name"),
            st.session_state.get("hub_username"),
            st.session_state.get("hub_token"),
        )
    ):
        st.success(f"Using the dataset repo {hub_username}/{project_name} on the Hub")

    st.sidebar.divider()

    st.sidebar.link_button("🧑‍🌾 New Project", DIBT_PARENT_APP_URL)
    
    if st.session_state["hub_token"] is None:
        st.error("Please provide a Hub token to generate answers")
        st.stop()

