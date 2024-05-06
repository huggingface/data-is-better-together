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

st.markdown("## 🧰 Data Generation Pipeline")

st.markdown(
    """
            Now we need to define the configuration for the pipeline that will generate the synthetic data.
            The pipeline will generate synthetic data by combining self-instruction and domain expert responses.
            The self-instruction step generates instructions based on seed terms, and the domain expert step generates \
            responses to those instructions. Take a look at the [distilabel docs](https://distilabel.argilla.io/latest/sections/learn/tasks/text_generation/#self-instruct) for more information.
            """
)

###############################################################
# INFERENCE
###############################################################

st.markdown("#### 🤖 Inference configuration")

st.write(
    """Add the url of the Huggingface inference API or endpoint that your pipeline should use to generate instruction and response pairs. \
    Some domain tasks may be challenging for smaller models, so you may need to iterate over your task definition and model selection. \
    This is a part of the process of generating high-quality synthetic data, human feedback is key to this process. \
    You can find compatible models here:"""
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

###############################################################
# PARAMETERS
###############################################################

st.divider()
st.markdown("#### 🧮 Parameters configuration")

st.write(
    "⚠️ Model and parameter choices significantly affect the quality of the generated data. \
    We reccomend that you start with generating a few samples and review the data. Then scale up from there. \
    You can run the pipeline multiple times with different configurations and append it to the same Argilla dataset."
)

st.markdown(
    "Number of generations are the samples that each model will generate for each seed term, \
    so if you have 10 seed terms, 2 instruction generations, and 2 response generations, you will have 40 samples in total."
)

self_intruct_num_generations = st.slider(
    "Number of generations for self-instruction", 1, 10, 2
)
domain_expert_num_generations = st.slider(
    "Number of generations for domain expert response", 1, 10, 2
)

with st.expander("🔥 Advanced parameters"):
    st.markdown(
        "Temperature is a hyperparameter that controls the randomness of the generated text. \
            Lower temperatures will generate more deterministic text, while higher temperatures \
            will add more variation to generations."
    )

    self_instruct_temperature = st.slider(
        "Temperature for self-instruction", 0.1, 1.0, 0.9
    )
    domain_expert_temperature = st.slider(
        "Temperature for domain expert", 0.1, 1.0, 0.9
    )

    st.markdown(
        "`max_new_tokens` is the maximum number of tokens (word like things) that can be generated by each model call. \
            This is a way to control the length of the generated text. in some cases, you may want to increase this to \
            generate longer responses. You should adapt this value to your model chice, but default of 2096 works \
            in most cases."
    )

    self_instruct_max_new_tokens = st.number_input(
        "Max new tokens for self-instruction", value=2096
    )
    domain_expert_max_new_tokens = st.number_input(
        "Max new tokens for domain expert", value=2096
    )

###############################################################
# ARGILLA API
###############################################################

st.divider()
st.markdown("#### 🔬 Argilla API details to push the generated dataset")
st.markdown(
    "Here you can define the Argilla API details to push the generated dataset to your Argilla space. \
        These are the defaults that were set up for the project. You can change them if needed."
)
argilla_url = st.text_input("Argilla API URL", ARGILLA_URL)
argilla_api_key = st.text_input("Argilla API Key", "owner.apikey")
argilla_dataset_name = st.text_input("Argilla Dataset Name", project_name)
st.divider()

###############################################################
# Pipeline Run
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
                "argilla_api_url": argilla_url,
                "argilla_dataset_name": argilla_dataset_name,
                "self_instruct_base_url": self_instruct_base_url,
                "domain_expert_base_url": domain_expert_base_url,
                "self_instruct_temperature": self_instruct_temperature,
                "domain_expert_temperature": domain_expert_temperature,
                "self_intruct_num_generations": self_intruct_num_generations,
                "domain_expert_num_generations": domain_expert_num_generations,
                "self_instruct_max_new_tokens": self_instruct_max_new_tokens,
                "domain_expert_max_new_tokens": domain_expert_max_new_tokens,
            },
            hub_username=hub_username,
            hub_token=hub_token,
            project_name=project_name,
        )

    st.success(
        f"Pipeline configuration pushed to the dataset repo {hub_username}/{project_name} on the Hub."
    )

    st.markdown(
        "To run the pipeline locally, you need to have the `distilabel` library installed. \
            You can install it using the following command:"
    )

    st.code(
        body="""
        # Install the distilabel library
        pip install distilabel
        """,
        language="bash",
    )

    st.markdown(
        "Next, you'll need to clone the pipeline code and install dependencies:"
    )

    st.code(
        """
        git clone https://github.com/huggingface/data-is-better-together
        cd data-is-better-together/domain-specific-datasets/distilabel_pipelines
        pip install -r requirements.txt
        huggingface-cli login
        """,
        language="bash",
    )

    st.markdown("Finally, you can run the pipeline using the following command:")

    st.code(
        f"""
        python domain_expert_pipeline.py {hub_username}/{project_name}""",
        language="bash",
    )
    st.markdown(
        "👩‍🚀 If you want to customise the pipeline take a look in `domain_expert_pipeline.py` \
            and the [distilabel docs](https://distilabel.argilla.io/)"
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
