# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from random import Random
from typing import Generic, Optional, Sequence, TypeVar

import chz
import pandas as pd
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    StopCondition,
)

from agentlightning import Dataset

T_task = TypeVar("T_task")

logger = logging.getLogger(__name__)


class AGLDummyEnv(Env, Generic[T_task]):

    def __init__(self, task: T_task) -> None:
        self.task = task

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        raise NotImplementedError("This method is not implemented for AGLDummyEnv")

    async def step(self, action: Action) -> StepResult:
        raise NotImplementedError("This method is not implemented for AGLDummyEnv")


class AGLDummyEnvGroupBuilder(EnvGroupBuilder, Generic[T_task]):

    def __init__(self, task: T_task, num_envs: int) -> None:
        self.task = task
        self.num_envs = num_envs

    async def make_envs(self) -> Sequence[AGLDummyEnv[T_task]]:
        return [AGLDummyEnv(self.task) for _ in range(self.num_envs)]


class AGLDataset(RLDataset, Generic[T_task]):
    """A dataset that produces batches of AGLDummyEnvGroupBuilder."""

    def __init__(
        self, dataset: Dataset[T_task], *, batch_size: int, shuffle: bool = True, group_size: int = 4, seed: int = 42
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_size = group_size

        if shuffle:
            self.indices = list(range(len(self.dataset)))
            Random(seed).shuffle(self.indices)
        else:
            self.indices = list(range(len(self.dataset)))

    def get_batch(self, index: int) -> Sequence[AGLDummyEnvGroupBuilder[T_task]]:
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.dataset))
        return [
            AGLDummyEnvGroupBuilder(self.dataset[self.indices[i]], self.group_size)
            for i in range(start_index, end_index)
        ]

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


@chz.chz
class AGLDatasetBuilder(RLDatasetBuilder, Generic[T_task]):

    batch_size: int
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    train_dataset: Optional[Dataset[T_task]] = None
    val_dataset: Optional[Dataset[T_task]] = None
    train_val_split: float = 0.7
    shuffle: bool = True
    group_size: int = 4
    seed: int = 42

    def _read_file(self, file: str) -> Dataset[T_task]:
        """Read a file and return a dataset.

        Supports parquet, csv and jsonl files.
        """
        if file.endswith(".parquet"):
            return pd.read_parquet(file).to_dict(orient="records")  # type: ignore
        elif file.endswith(".csv"):
            return pd.read_csv(file).to_dict(orient="records")  # type: ignore
        elif file.endswith(".jsonl"):
            return pd.read_json(file, lines=True).to_dict(orient="records")  # type: ignore
        else:
            raise ValueError(f"Unsupported file type: {file}")

    async def __call__(self) -> tuple[AGLDataset[T_task], AGLDataset[T_task]]:
        if self.train_file is not None:
            train_dataset = self._read_file(self.train_file)
        elif self.train_dataset is not None:
            train_dataset = self.train_dataset
        else:
            raise ValueError("No train dataset provided")

        if self.val_file is not None:
            val_dataset = self._read_file(self.val_file)
        elif self.val_dataset is not None:
            val_dataset = self.val_dataset
        else:
            indices = list(range(len(train_dataset)))
            Random(self.seed).shuffle(indices)
            val_indices = sorted(indices[int(len(indices) * self.train_val_split) :])
            train_indices = sorted(indices[: int(len(indices) * self.train_val_split)])
            logger.warning(
                "No validation dataset provided, splitting train dataset into train (%d) and validation (%d)",
                len(train_indices),
                len(val_indices),
            )
            train_dataset = [train_dataset[i] for i in train_indices]
            val_dataset = [train_dataset[i] for i in val_indices]

        return (
            AGLDataset(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                group_size=self.group_size,
                seed=self.seed,
            ),
            # For validation, always use batch_size=len(val_dataset) and group_size=1 to avoid dropping or repeating any samples
            AGLDataset(val_dataset, batch_size=len(val_dataset), shuffle=False, group_size=1),
        )
