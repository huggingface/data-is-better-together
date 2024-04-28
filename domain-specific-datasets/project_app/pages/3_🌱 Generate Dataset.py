import streamlit as st

from defaults import ARGILLA_URL
from hub import push_pipeline_params
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
st.subheader("Step 3. Run the pipeline to generate synthetic data")
st.write("Define the distilabel pipeline for generating the dataset.")

hub_username = st.session_state.get("hub_username")
project_name = st.session_state.get("project_name")
hub_token = st.session_state.get("hub_token")

###############################################################
# CONFIGURATION
###############################################################

st.divider()

st.markdown("## 🧰 Pipeline Configuration")

st.write(
    "Now we need to define the configuration for the pipeline that will generate the synthetic data."
)
st.write(
    "⚠️ Model and parameter choices significantly affect the quality of the generated data. \
    We reccomend that you start with generating a few samples and review the data. Then scale up from there. \
    You can run the pipeline multiple times with different configurations and append it to the same Argilla dataset."
)


st.markdown("#### 🤖 Inference configuration")

st.write(
    "Add the url of the Huggingface inference API or endpoint that your pipeline should use. You can find compatible models here:"
)

with st.expander("🤗 Recommended Models"):
    st.write("All inference endpoint compatible models can be found via the link below")
    st.link_button(
        "🤗 Inference compaptible models on the hub",
        "https://huggingface.co/models?pipeline_tag=text-generation&other=endpoints_compatible&sort=trending",
    )
    st.write("🔋Projects with sufficient resources could take advantage of LLama3 70b")
    st.code(
        "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    )

    st.write("🪫Projects with less resources could take advantage of LLama 3 8b")
    st.code(
        "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    )

    st.write("🍃Projects with even less resources could use Phi-3-mini-4k-instruct")
    st.code(
        "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    )

    st.write("Note Hugggingface Pro gives access to more compute resources")
    st.link_button(
        "🤗 Huggingface Pro",
        "https://huggingface.co/pricing",
    )


self_instruct_base_url = st.text_input(
    label="Model base URL for instruction generation",
    value="https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
)
domain_expert_base_url = st.text_input(
    label="Model base URL for domain expert response",
    value="https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
)

st.divider()
st.markdown("#### 🧮 Parameters configuration")

self_intruct_num_generations = st.slider(
    "Number of generations for self-instruction", 1, 10, 2
)
domain_expert_num_generations = st.slider(
    "Number of generations for domain expert response", 1, 10, 2
)
self_instruct_temperature = st.slider("Temperature for self-instruction", 0.1, 1.0, 0.9)
domain_expert_temperature = st.slider("Temperature for domain expert", 0.1, 1.0, 0.9)

st.divider()
st.markdown("#### 🔬 Argilla API details to push the generated dataset")
argilla_url = st.text_input("Argilla API URL", ARGILLA_URL)
argilla_api_key = st.text_input("Argilla API Key", "owner.apikey")
argilla_dataset_name = st.text_input("Argilla Dataset Name", project_name)
st.divider()

###############################################################
# LOCAL
###############################################################

st.markdown("## Run the pipeline")

st.markdown(
    "Once you've defined the pipeline configuration above, you can run the pipeline from your local machine."
)


if all(
    [
        argilla_api_key,
        argilla_url,
        self_instruct_base_url,
        domain_expert_base_url,
        self_intruct_num_generations,
        domain_expert_num_generations,
        self_instruct_temperature,
        domain_expert_temperature,
        hub_username,
        project_name,
        hub_token,
        argilla_dataset_name,
    ]
) and st.button("💾 Save Pipeline Config"):
    with st.spinner("Pushing pipeline to the Hub..."):
        push_pipeline_params(
            pipeline_params={
                "argilla_api_key": argilla_api_key,
                "argilla_api_url": argilla_url,
                "argilla_dataset_name": argilla_dataset_name,
                "self_instruct_base_url": self_instruct_base_url,
                "domain_expert_base_url": domain_expert_base_url,
                "self_instruct_temperature": self_instruct_temperature,
                "domain_expert_temperature": domain_expert_temperature,
                "self_intruct_num_generations": self_intruct_num_generations,
                "domain_expert_num_generations": domain_expert_num_generations,
            },
            hub_username=hub_username,
            hub_token=hub_token,
            project_name=project_name,
        )

    st.success(
        f"Pipeline configuration pushed to the dataset repo {hub_username}/{project_name} on the Hub."
    )

    st.markdown(
        "To run the pipeline locally, you need to have the `distilabel` library installed. You can install it using the following command:"
    )

    st.code(
        f"""
        
        # Install the distilabel library
        pip install distilabel
        """
    )

    st.markdown("Next, you'll need to clone your dataset repo and run the pipeline:")

    st.code(
        f"""
        git clone https://github.com/huggingface/data-is-better-together
        cd data-is-better-together/domain-specific-datasets/pipelines
        pip install -r requirements.txt
        """
    )

    st.markdown("Finally, you can run the pipeline using the following command:")

    st.code(
        f"""
        huggingface-cli login
        python domain_expert_pipeline.py {hub_username}/{project_name}""",
        language="bash",
    )
    st.markdown(
        "👩‍🚀 If you want to customise the pipeline take a look in `pipeline.py` and teh [distilabel docs](https://distilabel.argilla.io/)"
    )

    st.markdown(
        "🚀 Once you've run the pipeline your records will be available in the Argilla space"
    )

    st.link_button("🔗 Argilla Space", argilla_url)

    st.markdown("Once you've reviewed the data, you can publish it on the next page:")

    st.page_link(
        page="pages/4_🔍 Review Generated Data.py",
        label="Review Generated Data",
        icon="🔍",
    )

else:
    st.info("Please fill all the required fields.")
