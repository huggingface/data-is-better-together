import os

import streamlit as st

from defaults import (
    PROJECT_NAME,
    ARGILLA_URL,
    DIBT_PARENT_APP_URL,
    DATASET_URL,
    DATASET_REPO_ID,
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
        "Hub Token", type="password", value=os.environ.get("HF_TOKEN", None)
    )
    if st.session_state["hub_token"] is not None:
        os.environ["HF_TOKEN"] = st.session_state["hub_token"]
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


def create_seed_terms(topics: list[str], perspectives: list[str]) -> list[str]:
    """Create seed terms for self intruct to start from."""

    return [
        f"{topic} from a {perspective} perspective"
        for topic in topics
        for perspective in perspectives
    ]


def create_application_instruction(
    domain: str, system_prompt: str, examples: list[dict[str, str]]
) -> str:
    """Create the instruction for Self-Instruct task."""
    system_prompt = f"""AI assistant in the domain of {domain}. {system_prompt}"""
    examples_str = ""
    for example in examples:
        question = example["question"]
        answer = example["answer"]
        if len(answer) and len(question):
            examples_str += f"""\n- Question: {question}\n- Answer: {answer}\n"""
            examples_str += f"""\n- Question: {question}\n- Answer: {answer}\n"""
    if len(examples_str):
        system_prompt += """Below are some examples of questions and answers \
                            that the AI assistant would generate:"""
        system_prompt += "\nExamples:"
        system_prompt += f"\n{examples_str}"
    return system_prompt
