import json

import streamlit as st

from hub import push_dataset_to_hub, pull_seed_data_from_repo
from infer import query
from defaults import (
    N_PERSPECTIVES,
    N_TOPICS,
    SEED_DATA_PATH,
    PIPELINE_PATH,
    DATASET_REPO_ID,
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
# LOAD EXISTING DOMAIN DATA
################################################################################

DATASET_REPO_ID = (
    f"{st.session_state['hub_username']}/{st.session_state['project_name']}"
)
SEED_DATA = pull_seed_data_from_repo(
    DATASET_REPO_ID, hub_token=st.session_state["hub_token"]
)
DEFAULT_DOMAIN = SEED_DATA.get("domain", "")
DEFAULT_PERSPECTIVES = SEED_DATA.get("perspectives", [""])
DEFAULT_TOPICS = SEED_DATA.get("topics", [""])
DEFAULT_EXAMPLES = SEED_DATA.get("examples", [{"question": "", "answer": ""}])
DEFAULT_SYSTEM_PROMPT = SEED_DATA.get("domain_expert_prompt", "")

################################################################################
# Domain Expert Section
################################################################################

(
    tab_domain_expert,
    tab_domain_perspectives,
    tab_domain_topics,
    tab_examples,
    tab_raw_seed,
) = st.tabs(
    tabs=[
        "👩🏼‍🔬 Domain Expert",
        "🔍 Domain Perspectives",
        "🕸️ Domain Topics",
        "📚 Examples",
        "🌱 Raw Seed Data",
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
        [DEFAULT_PERSPECTIVES[0]],
    )
    perspectives_container = st.container()

    perspectives = [
        perspectives_container.text_input(
            f"Domain Perspective {i + 1}", value=perspective
        )
        for i, perspective in enumerate(perspectives)
    ]

    if st.button("Add Perspective", key="add_perspective"):
        n = len(perspectives)
        value = DEFAULT_PERSPECTIVES[n] if n < N_PERSPECTIVES else ""
        perspectives.append(
            perspectives_container.text_input(f"Domain Perspective {n + 1}", value="")
        )

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
        "topics",
        [DEFAULT_TOPICS[0]],
    )
    topics_container = st.container()
    topics = [
        topics_container.text_input(f"Domain Topic {i + 1}", value=topic)
        for i, topic in enumerate(topics)
    ]

    if st.button("Add Topic", key="add_topic"):
        n = len(topics)
        value = DEFAULT_TOPICS[n] if n < N_TOPICS else ""
        topics.append(topics_container.text_input(f"Domain Topics {n + 1}", value=""))

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

    examples = st.session_state.get(
        "examples",
        [
            {
                "question": "",
                "answer": "",
            }
        ],
    )

    for n, example in enumerate(examples, 1):
        question = example["question"]
        answer = example["answer"]
        examples_container = st.container()
        question_column, answer_column = examples_container.columns(2)

        if st.button(f"Generate Answer {n}"):
            if st.session_state["hub_token"] is None:
                st.error("Please provide a Hub token to generate answers")
            else:
                answer = query(question, st.session_state["hub_token"])
        with question_column:
            question = st.text_area(f"Question {n}", value=question)

        with answer_column:
            answer = st.text_area(f"Answer {n}", value=answer)
        examples[n - 1] = {"question": question, "answer": answer}
        st.session_state["examples"] = examples
        st.divider()

    if st.button("Add Example"):
        examples.append({"question": "", "answer": ""})
        st.session_state["examples"] = examples
        st.rerun()

################################################################################
# Save Domain Data
################################################################################

perspectives = list(filter(None, perspectives))
topics = list(filter(None, topics))

domain_data = {
    "domain": domain,
    "perspectives": perspectives,
    "topics": topics,
    "examples": examples,
    "domain_expert_prompt": domain_expert_prompt,
}

with open(SEED_DATA_PATH, "w") as f:
    json.dump(domain_data, f, indent=2)

with tab_raw_seed:
    st.code(json.dumps(domain_data, indent=2), language="json", line_numbers=True)

################################################################################
# Setup Dataset on the Hub
################################################################################

st.divider()


if st.button("🤗 Push Dataset Seed") and all(
    (
        domain,
        domain_expert_prompt,
        perspectives,
        topics,
        examples,
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

    push_dataset_to_hub(
        domain_seed_data_path=SEED_DATA_PATH,
        project_name=project_name,
        domain=domain,
        hub_username=hub_username,
        hub_token=hub_token,
        pipeline_path=PIPELINE_PATH,
    )

    st.success(
        f"Dataset seed created and pushed to the Hub. Check it out [here](https://huggingface.co/datasets/{hub_username}/{project_name})"
    )

    st.write("You can now move on to runnning your distilabel pipeline.")

    st.page_link(
        page="pages/3_🌱 Generate Dataset.py",
        label="Generate Dataset",
        icon="🌱",
    )

else:
    st.info(
        "Please fill in all the required domain fields to push the dataset seed to the Hub"
    )
