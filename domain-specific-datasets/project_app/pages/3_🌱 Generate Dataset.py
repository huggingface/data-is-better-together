import streamlit as st
from streamlit.errors import EntryNotFoundError

from hub import pull_seed_data_from_repo, push_pipeline_to_hub
from defaults import (
    DEFAULT_SYSTEM_PROMPT,
    PIPELINE_PATH,
    PROJECT_NAME,
    ARGILLA_URL,
    HUB_USERNAME,
    CODELESS_DISTILABEL,
)
from utils import project_sidebar

from pipeline import serialize_pipeline, run_pipeline, create_pipelines_run_command

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
st.subheader("Step 3. Run the pipeline to generate synthetic data")
st.write(
    "Define the project details, including the project name, domain, and API credentials"
)


###############################################################
# CONFIGURATION
###############################################################

st.divider()

st.markdown("### Pipeline Configuration")

st.write("🤗 Hub details to pull the seed data")
hub_username = st.text_input("Hub Username", HUB_USERNAME)
project_name = st.text_input("Project Name", PROJECT_NAME)
repo_id = f"{hub_username}/{project_name}"
hub_token = st.text_input("Hub Token", type="password")

st.write("🤖 Inference configuration")

st.write(
    "Add the url of the Huggingface inference API or endpoint that your pipeline should use. You can find compatible models here:"
)
st.link_button(
    "🤗 Inference compaptible models on the hub",
    "https://huggingface.co/models?pipeline_tag=text-generation&other=endpoints_compatible&sort=trending",
)

base_url = st.text_input("Base URL")

st.write("🔬 Argilla API details to push the generated dataset")
argilla_url = st.text_input("Argilla API URL", ARGILLA_URL)
argilla_api_key = st.text_input("Argilla API Key", "owner.apikey")
argilla_dataset_name = st.text_input("Argilla Dataset Name", project_name)
st.divider()

###############################################################
# LOCAL
###############################################################

st.markdown("### Run the pipeline")

st.write(
    "Once you've defined the pipeline configuration, you can run the pipeline from your local machine."
)

if CODELESS_DISTILABEL:
    st.write(
        """We recommend running the pipeline locally if you're planning on generating a large dataset. \
            But running the pipeline on this space is a handy way to get started quickly. Your synthetic
            samples will be pushed to Argilla and available for review.
            """
    )
    st.write(
        """If you're planning on running the pipeline on the space, be aware that it \
            will take some time to complete and you will need to maintain a \
            connection to the space."""
    )


if st.button("💻 Run pipeline locally", key="run_pipeline_local"):
    if all(
        [
            argilla_api_key,
            argilla_url,
            base_url,
            hub_username,
            project_name,
            hub_token,
            argilla_dataset_name,
        ]
    ):
        with st.spinner("Pulling seed data from the Hub..."):
            try:
                seed_data = pull_seed_data_from_repo(
                    repo_id=f"{hub_username}/{project_name}",
                    hub_token=hub_token,
                )
            except EntryNotFoundError:
                st.error(
                    "Seed data not found. Please make sure you pushed the data seed in Step 2."
                )

            domain = seed_data["domain"]
            perspectives = seed_data["perspectives"]
            topics = seed_data["topics"]
            examples = seed_data["examples"]
            domain_expert_prompt = seed_data["domain_expert_prompt"]

        with st.spinner("Serializing the pipeline configuration..."):
            serialize_pipeline(
                argilla_api_key=argilla_api_key,
                argilla_dataset_name=argilla_dataset_name,
                argilla_api_url=argilla_url,
                topics=topics,
                perspectives=perspectives,
                pipeline_config_path=PIPELINE_PATH,
                domain_expert_prompt=domain_expert_prompt or DEFAULT_SYSTEM_PROMPT,
                hub_token=hub_token,
                endpoint_base_url=base_url,
                examples=examples,
            )
            push_pipeline_to_hub(
                pipeline_path=PIPELINE_PATH,
                hub_token=hub_token,
                hub_username=hub_username,
                project_name=project_name,
            )

        st.success(f"Pipeline configuration saved to {hub_username}/{project_name}")

        st.info(
            "To run the pipeline locally, you need to have the `distilabel` library installed. You can install it using the following command:"
        )
        st.text(
            "Execute the following command to generate a synthetic dataset from the seed data:"
        )
        command_to_run = create_pipelines_run_command(
            hub_token=hub_token,
            pipeline_config_path=PIPELINE_PATH,
            argilla_dataset_name=argilla_dataset_name,
            argilla_api_key=argilla_api_key,
            argilla_api_url=argilla_url,
        )
        st.code(
            f"""
            pip install git+https://github.com/argilla-io/distilabel.git
            git clone https://huggingface.co/datasets/{hub_username}/{project_name}
            cd {project_name}
            pip install -r requirements.txt
            {' '.join(["python"] + command_to_run[1:])}
        """,
            language="bash",
        )
    else:
        st.error("Please fill all the required fields.")

###############################################################
# SPACE
###############################################################
if CODELESS_DISTILABEL:
    if st.button("🔥 Run pipeline right here, right now!"):
        if all(
            [
                argilla_api_key,
                argilla_url,
                base_url,
                hub_username,
                project_name,
                hub_token,
                argilla_dataset_name,
            ]
        ):
            with st.spinner("Pulling seed data from the Hub..."):
                try:
                    seed_data = pull_seed_data_from_repo(
                        repo_id=f"{hub_username}/{project_name}",
                        hub_token=hub_token,
                    )
                except EntryNotFoundError:
                    st.error(
                        "Seed data not found. Please make sure you pushed the data seed in Step 2."
                    )

                domain = seed_data["domain"]
                perspectives = seed_data["perspectives"]
                topics = seed_data["topics"]
                examples = seed_data["examples"]
                domain_expert_prompt = seed_data["domain_expert_prompt"]

                serialize_pipeline(
                    argilla_api_key=argilla_api_key,
                    argilla_dataset_name=argilla_dataset_name,
                    argilla_api_url=argilla_url,
                    topics=topics,
                    perspectives=perspectives,
                    pipeline_config_path=PIPELINE_PATH,
                    domain_expert_prompt=domain_expert_prompt or DEFAULT_SYSTEM_PROMPT,
                    hub_token=hub_token,
                    endpoint_base_url=base_url,
                    examples=examples,
                )

            with st.spinner("Starting the pipeline..."):
                logs = run_pipeline(
                    pipeline_config_path=PIPELINE_PATH,
                    argilla_api_key=argilla_api_key,
                    argilla_api_url=argilla_url,
                    hub_token=hub_token,
                    argilla_dataset_name=argilla_dataset_name,
                )

            st.success(f"Pipeline started successfully! 🚀")

            with st.expander(label="View Logs", expanded=True):
                for out in logs:
                    st.text(out)
        else:
            st.error("Please fill all the required fields.")
