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

st.markdown("# 🧑‍🌾 Domain Data Grower")
st.markdown("## 🌱 Create a dataset seed for aligning models to a specific domain")
st.markdown(
    "This app helps you create a dataset seed for building diverse domain-specific datasets for aligning models."
)
st.markdown(
    "Alignment datasets are used to fine-tune models to a specific domain or task, but as yet, there's a shortage of diverse datasets for this purpose."
)

project_name = st.text_input("Project Name", DEFAULT_DOMAIN)
domain = st.text_input("Domain", DEFAULT_DOMAIN)
hub_username = st.text_input("Hub Username", "argilla")
hub_token = st.text_input("Hub Token", type="password")
argilla_url = st.text_input("Argilla API URL", "https://argilla-farming.hf.space")
argilla_api_key = st.text_input("Argilla API Key", "owner.apikey")

st.header("👩🏼‍🔬 Domain Expert")
st.header("Define the domain expertise that you want to train a language model on.")
domain_expert_prompt = st.text_area(
    label="Domain Expertise",
    value=DEFAULT_SYSTEM_PROMPT,
)
st.header("👯️ Domain Perspectives")
st.markdown(
    "Perspectives are different viewpoints or angles from which a domain can be viewed. For example, the domain of farming can be viewed from the perspective of a commercial farmer or an independent family farmer."
)

perspectives = [
    st.text_input(f"Domain Perspective {n}", value=DEFAULT_PERSPECTIVES[n])
    for n in range(N_PERSPECTIVES)
]

st.header("🧺️ Domain Topics")
st.markdown(
    "Topics are the main themes or subjects that are relevant to the domain. For example, the domain of farming can have topics like soil health, crop rotation, or livestock management."
)
topics = [
    st.text_input(f"Domain Topic {n}", value=DEFAULT_TOPICS[n]) for n in range(N_TOPICS)
]

st.header("📚 Examples")

questions_answers = []

for n in range(N_EXAMPLES):
    default_question, default_answer = DEFAULT_EXAMPLES[n].values()
    st.subheader(f"Example {n + 1}")
    if st.button("Generate New Answer", key=f"generate_{n}"):
        default_answer = query(default_question)
    _question = st.text_area("Question", key=f"question_{n}", value=default_question)
    _answer = st.text_area("Answer", key=f"answer_{n}", value=default_answer)
    questions_answers.append((_question, _answer))


if st.button("Create Dataset Seed :seedling:"):
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

    push_dataset_to_hub(
        domain_seed_data_path=SEED_DATA_PATH,
        project_name=project_name,
        domain=domain,
        hub_username=hub_username,
        hub_token=hub_token,
        pipeline_path=PIPELINE_PATH,
    )

    st.markdown(
        f"# :seedling: Dataset is ready to grow \nDataset seed created and pushed to the Hub. Check it out [here](https://huggingface.co/{hub_username}/{project_name})"
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
