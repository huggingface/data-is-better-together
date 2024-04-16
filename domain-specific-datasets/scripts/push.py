import re
import datetime
import uuid

import argilla as rg
from datasets import Dataset, load_dataset
from dotenv import load_dotenv


load_dotenv()

rg.init(
    api_key="owner.apikey",
    api_url="https://argilla-farming.hf.space",
)

feedback_dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="instruction", use_markdown=True),
        rg.TextField(name="response", use_markdown=True),
        rg.TextField(name="task", use_markdown=True),
    ],
    questions=[
        rg.LabelQuestion(
            name="more",
            title="More like this?",
            labels=["Yes", "No"],
            type="label_selection",
        ),
        # rg.RatingQuestion(name="rating", values=[1, 2, 3, 4, 5]),
        # rg.TextQuestion(name="rationale", required=False),
        # rg.TextQuestion(name="improved_instruction", required=False),
        # rg.TextQuestion(name="improved_response", required=False),
        # rg.TextQuestion(name="improved_task", required=True),
    ],
)

feedback_dataset.add_vector_settings(
    rg.VectorSettings(name="instruction_vector", dimensions=384)
)
feedback_dataset.add_vector_settings(
    rg.VectorSettings(name="answer_vector", dimensions=384)
)
feedback_dataset.add_vector_settings(
    rg.VectorSettings(name="task_vector", dimensions=384)
)
feedback_dataset.add_vector_settings(
    rg.VectorSettings(name="rationale_vector", dimensions=384)
)


def build_record(
    instruction: str,
    answer: str,
    task: str,
    rationale: str = "No feedback provided",
    rating: int = 0,
    vectors: dict = None,
):

    try:
        rating = int(float(rating))
    except:
        rating = 1
    record = rg.FeedbackRecord(
        fields={"instruction": instruction, "response": answer, "task": task},
        vectors=vectors,
    )
    # record.suggestions = [
    #     {"question_name": "rating", "value": rating, "agent": "gpt-4"},
    #     {"question_name": "rationale", "value": rationale, "agent": "gpt-4"},
    # ]
    return record


def push_to_argilla(
    dataset: Dataset, name: str = str(uuid.uuid4()), workspace: str = "admin"
):
    feedback_records = []
    for _, row in dataset.to_pandas().iterrows():

        vectors = {col: row[col].tolist() for col in row.to_dict() if "vector" in col}

        record = build_record(
            instruction=row["instruction"],
            answer=row["answer"],
            rationale=row.get("rationale") or None,
            rating=row.get("rating") or 0,
            task=row["task"],
            vectors=vectors,
        )
        feedback_records.append(record)
    feedback_dataset.add_records(feedback_records)
    try:
        remote_dataset = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace)
        local_dataset = remote_dataset.pull()
        feedback_dataset.add_records(local_dataset.records)
    except Exception as e:
        print("Cannot pull from argilla")

    # strip timestamps from the dataset name with regex
    name = re.sub(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", name)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{name}-{now}"
    feedback_dataset.push_to_argilla(name=name, workspace=workspace)


def push_to_hub(name: str, workspace: str = "admin", repo_id: str = "burtenshaw"):
    feedback_dataset = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace)
    local_dataset = feedback_dataset.pull()
    local_dataset.push_to_huggingface(repo_id=repo_id)


if __name__ == "__main__":
    from argparse import ArgumentParser

    default_repo_id = "distilabel-internal-testing/farming-research-v0.2"
    default_dataset_split = "train"
    default_argilla_dataset_name = default_repo_id.replace("/", "_")
    default_argilla_workspace = "admin"

    parser = ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=default_repo_id)
    parser.add_argument("--name", type=str, default=default_argilla_dataset_name)
    parser.add_argument("--workspace", type=str, default=default_argilla_workspace)
    parser.add_argument("--split", type=str, default=default_dataset_split)
    args = parser.parse_args()

    dataset = load_dataset(args.repo_id)

    push_to_argilla(
        dataset=dataset[args.split],
        name="farming",
        workspace=args.workspace,
    )
