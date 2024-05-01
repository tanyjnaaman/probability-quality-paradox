# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================
"""Firefly (流萤) dataset for supervised instruction fine-tuning."""

from __future__ import annotations

from datasets import load_dataset
from src.packages.safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = ["FireflyDataset"]


class FireflyDataset(RawDataset):
    NAME: str = "firefly"

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or "YeungNLP/firefly-train-1.1M", split="train")

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(input=data["input"], answer=data["target"])

    def __len__(self) -> int:
        return len(self.data)
