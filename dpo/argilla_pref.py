# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
from typing import TYPE_CHECKING, List, Union

from typing_extensions import override

try:
    import argilla as rg
except ImportError:
    pass

from distilabel.steps.argilla.preference import PreferenceToArgilla
from distilabel.steps.base import StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class CustomPreferenceToArgilla(PreferenceToArgilla):
    metadata_properties: List[
        Union[
            rg.TermsMetadataProperty,  # type: ignore
            rg.FloatMetadataProperty,  # type: ignore
            rg.IntegerMetadataProperty,  # type: ignore
        ]
    ]

    def load(self) -> None:
        super().load()

        for metadata_property in self.metadata_properties:
            self._rg_dataset.add_metadata_property(metadata_property)  # type: ignore

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
                        metadata_property.name: input[metadata_property.name]
                        for metadata_property in self.metadata_properties
                        if metadata_property.name in input
                    },
                )
            )
        self._rg_dataset.add_records(records)  # type: ignore
        yield inputs
