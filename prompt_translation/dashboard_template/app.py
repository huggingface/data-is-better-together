from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import os
from typing import Dict, Tuple
from uuid import UUID

import altair as alt
import argilla as rg
from argilla.feedback import FeedbackDataset
from argilla.client.feedback.dataset.remote.dataset import RemoteFeedbackDataset
from huggingface_hub import restart_space
import gradio as gr
import pandas as pd

"""
This is the main file for the dashboard application. It contains the main function and the functions to obtain the data and create the charts.
It's designed as a template to recreate the dashboard for the prompt translation project of any language. 

To create a new dashboard, you need several environment variables, that you can easily set in the HuggingFace Space that you are using to host the dashboard:

- HF_TOKEN: Token with write access from your Hugging Face account: https://huggingface.co/settings/tokens
- SOURCE_DATASET: The dataset id of the source dataset
- SOURCE_WORKSPACE: The workspace id of the source dataset
- TARGET_RECORDS: The number of records that you have as a target to annotate. We usually set this to 500.
- ARGILLA_API_URL: Link to the Huggingface Space where the annotation effort is being hosted. For example, the Spanish one is https://somosnlp-dibt-prompt-translation-for-es.hf.space/
- ARGILLA_API_KEY: The API key to access the Huggingface Space. Please, write this as a secret in the Huggingface Space configuration.
"""

# Translation of legends and titles
ANNOTATED = "Annotations"
NUMBER_ANNOTATED = "Total Annotations"
PENDING = "Pending"

NUMBER_ANNOTATORS = "Number of annotators"
NAME = "Username"
NUMBER_ANNOTATIONS = "Number of annotations"

CATEGORY = "Category"


def restart() -> None:
    """
    This function restarts the space where the dashboard is hosted.
    """

    # Update Space name with your Space information
    gr.Info("Restarting space at " + str(datetime.datetime.now()))
    restart_space(
        "ignacioct/TryingRestartDashboard",
        token=os.getenv("HF_TOKEN"),
        # factory_reboot=True,
    )


def obtain_source_target_datasets() -> (
    Tuple[
        FeedbackDataset | RemoteFeedbackDataset, FeedbackDataset | RemoteFeedbackDataset
    ]
):
    """
    This function returns the source and target datasets to be used in the application.

    Returns:
        A tuple with the source and target datasets. The source dataset is filtered by the response status 'pending'.

    """

    # Obtain the public dataset and see how many pending records are there
    source_dataset = rg.FeedbackDataset.from_argilla(
        os.getenv("SOURCE_DATASET"), workspace=os.getenv("SOURCE_WORKSPACE")
    )
    filtered_source_dataset = source_dataset.filter_by(response_status=["pending"])

    # Obtain a list of users from the private workspace
    # target_dataset = rg.FeedbackDataset.from_argilla(
    #    os.getenv("RESULTS_DATASET"), workspace=os.getenv("RESULTS_WORKSPACE")
    # )

    target_dataset = source_dataset.filter_by(response_status=["submitted"])

    return filtered_source_dataset, target_dataset


def get_user_annotations_dictionary(
    dataset: FeedbackDataset | RemoteFeedbackDataset,
) -> Dict[str, int]:
    """
    This function returns a dictionary with the username as the key and the number of annotations as the value.

    Args:
        dataset: The dataset to be analyzed.
    Returns:
        A dictionary with the username as the key and the number of annotations as the value.
    """
    output = {}
    for record in dataset:
        for response in record.responses:
            if str(response.user_id) not in output.keys():
                output[str(response.user_id)] = 1
            else:
                output[str(response.user_id)] += 1

    # Changing the name of the keys, from the id to the username
    for key in list(output.keys()):
        output[rg.User.from_id(UUID(key)).username] = output.pop(key)

    return output


def donut_chart_total() -> alt.Chart:
    """
    This function returns a donut chart with the progress of the total annotations.
    Counts each record that has been annotated at least once.

    Returns:
        An altair chart with the donut chart.
    """

    # Load your data
    annotated_records = len(target_dataset)
    pending_records = int(os.getenv("TARGET_RECORDS")) - annotated_records

    # Prepare data for the donut chart
    source = pd.DataFrame(
        {
            "values": [annotated_records, pending_records],
            "category": [ANNOTATED, PENDING],
            "colors": [
                "#4682b4",
                "#e68c39",
            ],  # Blue for Completed, Orange for Remaining
        }
    )

    domain = source["category"].tolist()
    range_ = source["colors"].tolist()

    base = alt.Chart(source).encode(
        theta=alt.Theta("values:Q", stack=True),
        radius=alt.Radius(
            "values", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)
        ),
        color=alt.Color(
            field="category",
            type="nominal",
            scale=alt.Scale(domain=domain, range=range_),
            legend=alt.Legend(title=CATEGORY),
        ),
    )

    c1 = base.mark_arc(innerRadius=20, stroke="#fff")

    c2 = base.mark_text(radiusOffset=20).encode(text="values:Q")

    chart = c1 + c2

    return chart


def kpi_chart_remaining() -> alt.Chart:
    """
    This function returns a KPI chart with the remaining amount of records to be annotated.
    Returns:
        An altair chart with the KPI chart.
    """

    pending_records = int(os.getenv("TARGET_RECORDS")) - len(target_dataset)
    # Assuming you have a DataFrame with user data, create a sample DataFrame
    data = pd.DataFrame({"Category": [PENDING], "Value": [pending_records]})

    # Create Altair chart
    chart = (
        alt.Chart(data)
        .mark_text(fontSize=100, align="center", baseline="middle", color="#e68b39")
        .encode(text="Value:N")
        .properties(title=PENDING, width=250, height=200)
    )

    return chart


def kpi_chart_submitted() -> alt.Chart:
    """
    This function returns a KPI chart with the total amount of records that have been annotated.
    Returns:
        An altair chart with the KPI chart.
    """

    total = len(target_dataset)

    # Assuming you have a DataFrame with user data, create a sample DataFrame
    data = pd.DataFrame({"Category": [NUMBER_ANNOTATED], "Value": [total]})

    # Create Altair chart
    chart = (
        alt.Chart(data)
        .mark_text(fontSize=100, align="center", baseline="middle", color="steelblue")
        .encode(text="Value:N")
        .properties(title=NUMBER_ANNOTATED, width=250, height=200)
    )

    return chart


def kpi_chart_total_annotators() -> alt.Chart:
    """
    This function returns a KPI chart with the total amount of annotators.

    Returns:
        An altair chart with the KPI chart.
    """

    # Obtain the total amount of annotators
    total_annotators = len(user_ids_annotations)

    # Assuming you have a DataFrame with user data, create a sample DataFrame
    data = pd.DataFrame({"Category": [NUMBER_ANNOTATORS], "Value": [total_annotators]})

    # Create Altair chart
    chart = (
        alt.Chart(data)
        .mark_text(fontSize=100, align="center", baseline="middle", color="steelblue")
        .encode(text="Value:N")
        .properties(title=NUMBER_ANNOTATORS, width=250, height=200)
    )

    return chart


def render_hub_user_link(hub_id: str) -> str:
    """
    This function returns a link to the user's profile on Hugging Face.

    Args:
        hub_id: The user's id on Hugging Face.

    Returns:
        A string with the link to the user's profile on Hugging Face.
    """
    link = f"https://huggingface.co/{hub_id}"
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{hub_id}</a>'


def obtain_top_users(user_ids_annotations: Dict[str, int], N: int = 50) -> pd.DataFrame:
    """
    This function returns the top N users with the most annotations.

    Args:
        user_ids_annotations: A dictionary with the user ids as the key and the number of annotations as the value.

    Returns:
        A pandas dataframe with the top N users with the most annotations.
    """

    dataframe = pd.DataFrame(
        user_ids_annotations.items(), columns=[NAME, NUMBER_ANNOTATIONS]
    )
    dataframe[NAME] = dataframe[NAME].apply(render_hub_user_link)
    dataframe = dataframe.sort_values(by=NUMBER_ANNOTATIONS, ascending=False)
    return dataframe.head(N)


def fetch_data() -> None:
    """
    This function fetches the data from the source and target datasets and updates the global variables.
    """

    print(f"Starting to fetch data: {datetime.datetime.now()}")

    global source_dataset, target_dataset, user_ids_annotations, annotated, remaining, percentage_completed, top_dataframe
    source_dataset, target_dataset = obtain_source_target_datasets()
    user_ids_annotations = get_user_annotations_dictionary(target_dataset)

    annotated = len(target_dataset)
    remaining = int(os.getenv("TARGET_RECORDS")) - annotated
    percentage_completed = round(
        (annotated / int(os.getenv("TARGET_RECORDS"))) * 100, 1
    )

    # Print the current date and time
    print(f"Data fetched: {datetime.datetime.now()}")


def get_top(N=50) -> pd.DataFrame:
    """
    This function returns the top N users with the most annotations.

    Args:
        N: The number of users to be returned. 50 by default

    Returns:
        A pandas dataframe with the top N users with the most annotations.
    """

    return obtain_top_users(user_ids_annotations, N=N)


def main() -> None:

    # Connect to the space with rg.init()
    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY"),
    )

    # Fetch the data initially
    fetch_data()

    # To avoid the orange border for the Gradio elements that are in constant loading
    css = """
    .generating {
        border: none;
    }
    """

    with gr.Blocks(css=css, delete_cache=(300, 300)) as demo:
        gr.Markdown(
            """
            # 🌍 [YOUR LANGUAGE] - Multilingual Prompt Evaluation Project

            Hugging Face and @argilla are developing [Multilingual Prompt Evaluation Project](https://github.com/huggingface/data-is-better-together/tree/main/prompt_translation) project. It is an open multilingual benchmark for evaluating language models, and of course, also for [YOUR LANGUAGE].

            ## The goal is to translate 500 Prompts
            And as always: data is needed for that! The community selected the best 500 prompts that will form the benchmark. In English, of course.
            **That's why we need your help**: if we all translate the 500 prompts, we can add [YOUR LANGUAGE] to the leaderboard.

            ## How to participate
            Participating is easy. Go to the [annotation space][add a link to your annotation dataset], log in or create a Hugging Face account, and you can start working.
            Thanks in advance! Oh, and we'll give you a little push: GPT4 has already prepared a translation suggestion for you.
            """
        )

        gr.Markdown(
            f"""
            ## 🚀 Current Progress
            This is what we've achieved so far!
            """
        )
        with gr.Row():

            kpi_submitted_plot = gr.Plot(label="Plot")
            demo.load(
                kpi_chart_submitted,
                inputs=[],
                outputs=[kpi_submitted_plot],
            )

            kpi_remaining_plot = gr.Plot(label="Plot")
            demo.load(
                kpi_chart_remaining,
                inputs=[],
                outputs=[kpi_remaining_plot],
            )

            donut_total_plot = gr.Plot(label="Plot")
            demo.load(
                donut_chart_total,
                inputs=[],
                outputs=[donut_total_plot],
            )

        gr.Markdown(
            """
            ## 👾 Hall of Fame
            Here you can see the top contributors and the number of annotations they have made.
            """
        )

        with gr.Row():

            kpi_hall_plot = gr.Plot(label="Plot")
            demo.load(kpi_chart_total_annotators, inputs=[], outputs=[kpi_hall_plot])

            top_df_plot = gr.Dataframe(
                headers=[NAME, NUMBER_ANNOTATIONS],
                datatype=[
                    "markdown",
                    "number",
                ],
                row_count=50,
                col_count=(2, "fixed"),
                interactive=False,
            )
            demo.load(get_top, None, [top_df_plot])

    # Manage background refresh
    scheduler = BackgroundScheduler()
    _ = scheduler.add_job(restart, "interval", minutes=30)
    scheduler.start()

    # Launch the Gradio interface
    demo.launch()


if __name__ == "__main__":
    main()
