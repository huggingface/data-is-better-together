import json

import streamlit as st

from hub import push_dataset_to_hub
from infer import query
from defaults import (
    DEFAULT_DOMAIN,
    DEFAULT_PERSPECTIVES,
    DEFAULT_TOPICS,
    DEFAULT_EXAMPLES,
    DEFAULT_SYSTEM_PROMPT,
    N_PERSPECTIVES,
    N_TOPICS,
    SEED_DATA_PATH,
    PIPELINE_PATH,
)
from utils import project_sidebar

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
st.subheader(
    "Step 2. Define the specific domain that you want to generate synthetic data for.",
)
st.write(
    "Define the project details, including the project name, domain, and API credentials"
)

################################################################################
# Domain Expert Section
################################################################################

(
    tab_domain_expert,
    tab_domain_perspectives,
    tab_domain_topics,
    tab_examples,
) = st.tabs(
    tabs=[
        "👩🏼‍🔬 Domain Expert",
        "🔍 Domain Perspectives",
        "🕸️ Domain Topics",
        "📚 Examples",
    ]
)

with tab_domain_expert:
    st.text("Define the domain expertise that you want to train a language model")
    st.info(
        "A domain expert is a person who is an expert in a particular field or area. For example, a domain expert in farming would be someone who has extensive knowledge and experience in farming and agriculture."
    )

    domain = st.text_input("Domain Name", DEFAULT_DOMAIN)

    domain_expert_prompt = st.text_area(
        label="Domain Expert Definition",
        value=DEFAULT_SYSTEM_PROMPT,
        height=200,
    )

################################################################################
# Domain Perspectives
################################################################################

with tab_domain_perspectives:
    st.text("Define the different perspectives from which the domain can be viewed")
    st.info(
        """
    Perspectives are different viewpoints or angles from which a domain can be viewed. 
    For example, the domain of farming can be viewed from the perspective of a commercial 
    farmer or an independent family farmer."""
    )

    perspectives = st.session_state.get(
        "perspectives",
        [st.text_input(f"Domain Perspective 0", value=DEFAULT_PERSPECTIVES[0])],
    )

    if st.button("Add New Perspective"):
        n = len(perspectives)
        value = DEFAULT_PERSPECTIVES[n] if n < N_PERSPECTIVES else ""
        perspectives.append(st.text_input(f"Domain Perspective {n}", value=""))
        st.session_state["perspectives"] = perspectives


################################################################################
# Domain Topics
################################################################################

with tab_domain_topics:
    st.text("Define the main themes or subjects that are relevant to the domain")
    st.info(
        """Topics are the main themes or subjects that are relevant to the domain. For example, the domain of farming can have topics like soil health, crop rotation, or livestock management."""
    )
    topics = st.session_state.get(
        "topics", [st.text_input(f"Domain Topic 0", value=DEFAULT_TOPICS[0])]
    )
    new_topic = st.button("Add New Topic")

    if new_topic:
        n = len(topics)
        value = DEFAULT_TOPICS[n] if n < N_TOPICS else ""
        topics.append(st.text_input(f"Domain Topic {n}", value=value))
        st.session_state["topics"] = topics


################################################################################
# Examples Section
################################################################################

with tab_examples:
    st.text(
        "Add high-quality questions and answers that can be used to generate synthetic data"
    )
    st.info(
        """
    Examples are high-quality questions and answers that can be used to generate 
    synthetic data for the domain. These examples will be used to train the language model
    to generate questions and answers.
    """
    )

    questions_answers = st.session_state.get(
        "questions_answers",
        [
            (
                st.text_area(
                    "Question", key="question_0", value=DEFAULT_EXAMPLES[0]["question"]
                ),
                st.text_area(
                    "Answer", key="answer_0", value=DEFAULT_EXAMPLES[0]["answer"]
                ),
            )
        ],
    )

    if st.button("Add New Example"):
        n = len(questions_answers)
        default_question, default_answer = DEFAULT_EXAMPLES[n].values()
        st.subheader(f"Example {n + 1}")
        if st.button("Generate New Answer", key=f"generate_{n}"):
            default_answer = query(default_question)
        _question = st.text_area(
            "Question", key=f"question_{n}", value=default_question
        )
        _answer = st.text_area("Answer", key=f"answer_{n}", value=default_answer)
        questions_answers.append((_question, _answer))
        st.session_state["questions_answers"] = questions_answers

################################################################################
# Setup Dataset on the Hub
################################################################################

st.divider()


with st.expander("🤗 Repository Details"):
    st.write("Define the dataset repo details on the Hub")
    st.session_state["project_name"] = st.text_input("Project Name", None)
    st.session_state["hub_username"] = st.text_input("Hub Username", None)
    st.session_state["hub_token"] = st.text_input(
        "Hub Token", type="password", value=None
    )

    if all(
        (
            st.session_state.get("project_name"),
            st.session_state.get("hub_username"),
            st.session_state.get("hub_token"),
        )
    ):
        st.success(f"Using the dataset repo {hub_username}/{project_name} on the Hub")


if st.button("🤗 Push Dataset Seed") and all(
    (
        domain,
        domain_expert_prompt,
        perspectives,
        topics,
        questions_answers,
    )
):
    if all(
        (
            st.session_state.get("project_name"),
            st.session_state.get("hub_username"),
            st.session_state.get("hub_token"),
        )
    ):
        project_name = st.session_state["project_name"]
        hub_username = st.session_state["hub_username"]
        hub_token = st.session_state["hub_token"]
    else:
        st.error(
            "Please create a dataset repo on the Hub before pushing the dataset seed"
        )
        st.stop()

    perspectives = list(filter(None, perspectives))
    topics = list(filter(None, topics))
    examples = [{"question": q, "answer": a} for q, a in questions_answers]

    domain_data = {
        "domain": domain,
        "perspectives": perspectives,
        "topics": topics,
        "examples": examples,
        "domain_expert_prompt": domain_expert_prompt,
    }

    with open(SEED_DATA_PATH, "w") as f:
        json.dump(domain_data, f, indent=2)

    push_dataset_to_hub(
        domain_seed_data_path=SEED_DATA_PATH,
        project_name=project_name,
        domain=domain,
        hub_username=hub_username,
        hub_token=hub_token,
        pipeline_path=PIPELINE_PATH,
    )

    st.sidebar.success(
        f"Dataset seed created and pushed to the Hub. Check it out [here](https://huggingface.co/datasets/{hub_username}/{project_name})"
    )
else:
    st.info(
        "Please fill in all the required domain fields to push the dataset seed to the Hub"
    )
