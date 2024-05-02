import hashlib
from typing import TYPE_CHECKING, Any, Dict, List

import argilla as rg
from distilabel.steps import PreferenceToArgilla, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import (
        RatingQuestion,
        StepOutput,
        TextQuestion,
    )
    from typing import Union
    from typing_extensions import override


class CustomPreferenceToArgilla(PreferenceToArgilla):
    """
    Custom PreferenceToArgilla step that adds metadata properties to the feedback records.
    This allows filtering based on metadata properties in the Argilla UI.
    """

    metadata_properties: List[Dict[str, Any]]

    def load(self) -> None:
        super().load()
        for metadata_property in self.metadata_properties:
            metadata_property_type = metadata_property.pop("type", None)
            if metadata_property_type == "float":
                metadata_property = rg.FloatMetadataProperty.parse_obj(
                    metadata_property
                )
            elif metadata_property_type == "integer":
                metadata_property = rg.IntegerMetadataProperty.parse_obj(
                    metadata_property
                )
            elif metadata_property_type == "terms":
                metadata_property = rg.TermsMetadataProperty.parse_obj(
                    metadata_property
                )
            else:
                break
            self._rg_dataset.add_metadata_property(metadata_property)  # type: ignore

    def _rating_rationale_pairs(
        self,
    ) -> List[Union["RatingQuestion", "TextQuestion"]]:
        questions = super()._rating_rationale_pairs()
        questions.append(
            rg.TextQuestion(  # type: ignore
                name="improved_response",
                title="How would you improve the response?",
                required=False,
            )
        )
        return questions

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        records = []
        for input in inputs:
            # Generate the SHA-256 hash of the instruction to use it as the metadata
            instruction_id = hashlib.sha256(
                input["instruction"].encode("utf-8")  # type: ignore
            ).hexdigest()

            generations = {
                f"{self._generations}-{idx}": generation
                for idx, generation in enumerate(input["generations"])  # type: ignore
            }
            records.append(  # type: ignore
                rg.FeedbackRecord(  # type: ignore
                    fields={
                        "id": instruction_id,
                        "instruction": input["instruction"],  # type: ignore
                        **generations,
                    },
                    suggestions=self._add_suggestions_if_any(input),  # type: ignore
                    metadata={
                        metadata_property["name"]: input[metadata_property["name"]]
                        for metadata_property in self.metadata_properties
                        if metadata_property["name"] in input
                    },
                )
            )
        self._rg_dataset.add_records(records)  # type: ignore
        yield inputs
