import streamlit as st

from hub import pull_seed_data_from_repo
from defaults import (
    DEFAULT_DOMAIN,
    DEFAULT_SYSTEM_PROMPT,
    PIPELINE_PATH,
)
from pipeline import serialize_pipeline, run_pipeline

st.set_page_config(
    page_title="Domain Data Grower",
    page_icon="🧑‍🌾",
)

################################################################################
# HEADER
################################################################################

st.header("🧑‍🌾 Domain Data Grower")
st.divider()
st.subheader("Step 3. Run the pipeline to generate synthetic data")
st.write(
    "Define the project details, including the project name, domain, and API credentials"
)


###############################################################
# CONFIGURATION
###############################################################


st.divider()
st.write("Hub details to pull the seed data")
hub_username, project_name = st.text_input(
    "Dataset repo id", value="argilla/farming"
).split("/")
hub_token = st.text_input("Hub Token", type="password")

st.divider()
st.write("Inference configuration")

base_url = st.text_input("Base URL")


st.divider()
st.write("Argilla API details to push the generated dataset")
argilla_url = st.text_input("Argilla API URL", "https://argilla-farming.hf.space")
argilla_api_key = st.text_input("Argilla API Key", "owner.apikey")

st.divider()

###############################################################
# LOCAL
###############################################################

if st.button("💻 Run pipeline locally", key="run_pipeline_local"):
    if all(
        [
            argilla_api_key,
            argilla_url,
            base_url,
            hub_username,
            project_name,
            hub_token,
        ]
    ):
        seed_data = pull_seed_data_from_repo(
            repo_id=f"{hub_username}/{project_name}",
            hub_token=hub_token,
        )

        domain = seed_data["domain"]
        perspectives = seed_data["perspectives"]
        topics = seed_data["topics"]
        examples = seed_data["examples"]
        domain_expert_prompt = seed_data["domain_expert_prompt"]

        serialize_pipeline(
            argilla_api_key=argilla_api_key,
            argilla_dataset_name=project_name or DEFAULT_DOMAIN,
            argilla_api_url=argilla_url,
            topics=topics,
            perspectives=perspectives,
            pipeline_config_path=PIPELINE_PATH,
            domain_expert_prompt=domain_expert_prompt or DEFAULT_SYSTEM_PROMPT,
            hub_token=hub_token,
            endpoint_base_url=base_url,
            examples=examples,
        )
        st.success(f"Pipeline configuration saved to {PIPELINE_PATH}")

        st.info(
            "To run the pipeline locally, you need to have the `distilabel` library installed. You can install it using the following command:"
        )
        st.text(
            "Execute the following command to generate a synthetic dataset from the seed data:"
        )
        st.code(
            f"""
            pip install git+https://github.com/argilla-io/distilabel.git
            git clone https://huggingface.co/{hub_username}/{project_name}
            cd {project_name}
            distilabel pipeline run --config pipeline.yaml
        """
        )
    else:
        st.error("Please fill all the required fields.")

###############################################################
# SPACE
###############################################################

if st.button("🔥 Run pipeline right here, right now!"):
    if all(
        [
            argilla_api_key,
            argilla_url,
            base_url,
            hub_username,
            project_name,
            hub_token,
        ]
    ):
        seed_data = pull_seed_data_from_repo(
            repo_id=f"{hub_username}/{project_name}",
            hub_token=hub_token,
        )

        domain = seed_data["domain"]
        perspectives = seed_data["perspectives"]
        topics = seed_data["topics"]
        examples = seed_data["examples"]
        domain_expert_prompt = seed_data["domain_expert_prompt"]

        serialize_pipeline(
            argilla_api_key=argilla_api_key,
            argilla_dataset_name=project_name or DEFAULT_DOMAIN,
            argilla_api_url=argilla_url,
            topics=topics,
            perspectives=perspectives,
            pipeline_config_path=PIPELINE_PATH,
            domain_expert_prompt=domain_expert_prompt or DEFAULT_SYSTEM_PROMPT,
            hub_token=hub_token,
            endpoint_base_url=base_url,
            examples=examples,
        )
        st.success(f"Pipeline configuration saved to {PIPELINE_PATH}")

        logs = run_pipeline(PIPELINE_PATH)

        st.success(f"Running the pipeline.")

        with st.expander("View Logs"):
            for out in logs:
                st.text(out)
    else:
        st.error("Please fill all the required fields.")
