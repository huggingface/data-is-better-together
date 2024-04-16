import json

from pytest import mark
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
    N_EXAMPLES,
    SEED_DATA_PATH,
    PIPELINE_PATH,
)
from pipeline import serialize_pipeline

################################################################################
# Introduction and Project Setup
################################################################################

introduction = """# 🧑‍🌾 Domain Data Grower
## 🌱 Create a dataset seed for aligning models to a specific domain

This app helps you create a dataset seed for building diverse domain-specific datasets for aligning models.
Alignment datasets are used to fine-tune models to a specific domain or task, but as yet, there's a shortage of diverse datasets for this purpose.
"""
st.markdown(introduction)

project_name = st.text_input("Project Name", DEFAULT_DOMAIN)
domain = st.text_input("Domain", DEFAULT_DOMAIN)
hub_username = st.text_input("Hub Username", "argilla")
hub_token = st.text_input("Hub Token", type="password")
argilla_url = st.text_input("Argilla API URL", "https://argilla-farming.hf.space")
argilla_api_key = st.text_input("Argilla API Key", "owner.apikey")

################################################################################
# Domain Expert Section
################################################################################
st.divider()
st.header("👩🏼‍🔬 Domain Expert")
st.text("Define the domain expertise that you want to train a language model")
st.info(
    "A domain expert is a person who is an expert in a particular field or area. For example, a domain expert in farming would be someone who has extensive knowledge and experience in farming and agriculture."
)

domain_expert_prompt = st.text_area(
    label="Domain Expert Definition",
    value=DEFAULT_SYSTEM_PROMPT,
    height=200,
)

################################################################################
# Domain Perspectives
################################################################################

st.divider()
st.header("🔍 Domain Perspectives")
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
new_perspective = st.button("Add New Perspective")

if new_perspective:
    n = len(perspectives)
    value = DEFAULT_PERSPECTIVES[n] if n < N_PERSPECTIVES else ""
    perspectives.append(st.text_input(f"Domain Perspective {n}", value=""))
    st.session_state["perspectives"] = perspectives


################################################################################
# Domain Topics
################################################################################

st.divider()
st.header("🕸️ Domain Topics")
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

st.divider()
st.header("📚 Examples")
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
            st.text_area("Answer", key="answer_0", value=DEFAULT_EXAMPLES[0]["answer"]),
        )
    ],
)

new_question = st.button("Add New Example")

if new_question:
    n = len(questions_answers)
    default_question, default_answer = DEFAULT_EXAMPLES[n].values()
    st.subheader(f"Example {n + 1}")
    if st.button("Generate New Answer", key=f"generate_{n}"):
        default_answer = query(default_question)
    _question = st.text_area("Question", key=f"question_{n}", value=default_question)
    _answer = st.text_area("Answer", key=f"answer_{n}", value=default_answer)
    questions_answers.append((_question, _answer))
    st.session_state["questions_answers"] = questions_answers

################################################################################
# Create Dataset Seed
################################################################################

create_text = """
# 🤗 Create Dataset on the Hub
Create a dataset seed and push it to the Hub to grow a domain-specific dataset.
"""
st.markdown(create_text)
if st.button("Create Dataset Seed :seedling:"):
    ############################################################
    # Create Dataset Seed and serialize pipeline
    ############################################################

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

    serialize_pipeline(
        argilla_api_key=argilla_api_key,
        argilla_dataset_name=project_name or DEFAULT_DOMAIN,
        argilla_api_url=argilla_url,
        topics=topics,
        perspectives=perspectives,
        pipeline_config_path=PIPELINE_PATH,
        domain_expert_prompt=domain_expert_prompt or DEFAULT_SYSTEM_PROMPT,
    )

    ############################################################
    # Setup Dataset on the Hub
    ############################################################

    push_dataset_to_hub(
        domain_seed_data_path=SEED_DATA_PATH,
        project_name=project_name,
        domain=domain,
        hub_username=hub_username,
        hub_token=hub_token,
        pipeline_path=PIPELINE_PATH,
    )

    st.success(
        f"Dataset seed created and pushed to the Hub. Check it out [here](https://huggingface.co/{hub_username}/{project_name})"
    )

    instructions = f"""
        Execute the following command to generate a synthetic dataset from the seed data:
        ```bash
        git clone https://huggingface.co/{hub_username}/{project_name}
        cd {project_name}
        distilabel pipeline run --config pipeline.yaml
        ```
    """

    st.markdown(instructions)
