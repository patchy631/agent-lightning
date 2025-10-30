# Copyright (c) Microsoft. All rights reserved.

"""Environment and dataset utilities for Agent-lightning with Tinker.

This module provides dataset and environment wrappers that bridge Agent-lightning's
task-based approach with Tinker's RL training infrastructure.
"""

from __future__ import annotations

import logging
from random import Random
from typing import Generic, List, Optional, Sequence, TypeVar

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
    """Placeholder environment for Agent-lightning tasks.

    Since Agent-lightning handles environment logic within user agents,
    this class serves as a placeholder to satisfy Tinker's interface requirements.

    Attributes:
        task: The task data for this environment instance.
    """

    def __init__(self, task: T_task) -> None:
        """Initialize the dummy environment with a task.

        Args:
            task: The task data for this environment instance.
        """
        self.task = task

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        raise NotImplementedError("This method is not implemented for AGLDummyEnv")

    async def step(self, action: Action) -> StepResult:
        raise NotImplementedError("This method is not implemented for AGLDummyEnv")


class AGLDummyEnvGroupBuilder(EnvGroupBuilder, Generic[T_task]):
    """Builder for creating groups of AGLDummyEnv instances.

    Attributes:
        task: The task to use for all environments in the group.
        num_envs: Number of environments to create in the group.
    """

    def __init__(self, task: T_task, num_envs: int) -> None:
        """Initialize the environment group builder.

        Args:
            task: The task to use for all environments.
            num_envs: Number of environments to create.
        """
        self.task = task
        self.num_envs = num_envs

    async def make_envs(self) -> Sequence[AGLDummyEnv[T_task]]:
        """Create a sequence of dummy environments.

        Returns:
            Sequence of AGLDummyEnv instances.
        """
        return [AGLDummyEnv(self.task) for _ in range(self.num_envs)]


class AGLDataset(RLDataset, Generic[T_task]):
    """RL dataset that produces batches of AGLDummyEnvGroupBuilder instances.

    This dataset manages Agent-lightning task data for Tinker's RL training loop,
    handling batching, shuffling, and multi-epoch iteration.

    When enabling shuffling with multi-epochs, the dataset will be individually shuffled for each epoch.
    So the remainder of the dataset length divided by batch size will be dropped for each epoch.

    Attributes:
        dataset: The underlying Agent-lightning dataset of tasks.
        batch_size: Number of tasks per batch.
        shuffle: Whether to shuffle the dataset each epoch.
        group_size: Number of rollouts per task group.
        n_epochs: Number of training epochs.
        indices: Flattened list of dataset indices across all epochs.
    """

    def __init__(
        self,
        dataset: Dataset[T_task],
        *,
        batch_size: int,
        shuffle: bool = True,
        group_size: int = 4,
        seed: int = 42,
        n_epochs: int = 1,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset: Agent-lightning dataset of tasks.
            batch_size: Number of tasks per batch.
            shuffle: Whether to shuffle the dataset each epoch.
            group_size: Number of rollouts per task group.
            seed: Random seed for shuffling.
            n_epochs: Number of training epochs.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_size = group_size
        self.n_epochs = n_epochs

        self.indices: List[int] = []
        if shuffle:
            random_state = Random(seed)
            for _ in range(n_epochs):
                indices = list(range(len(self.dataset)))
                random_state.shuffle(indices)
                # Drop last for each epoch
                self.indices.extend(indices[: len(indices) - len(indices) % self.batch_size])
        else:
            for _ in range(n_epochs):
                self.indices.extend(list(range(len(self.dataset))))

    def get_batch(self, index: int) -> Sequence[AGLDummyEnvGroupBuilder[T_task]]:
        """Get a batch of environment group builders.

        Args:
            index: Batch index.

        Returns:
            Sequence of AGLDummyEnvGroupBuilder instances for the batch.
        """
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.indices))
        return [
            AGLDummyEnvGroupBuilder(self.dataset[self.indices[i]], self.group_size)
            for i in range(start_index, end_index)
        ]

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size


@chz.chz
class AGLDatasetBuilder(RLDatasetBuilder, Generic[T_task]):
    """Builder for creating train/val AGLDataset splits.

    Supports loading datasets from files (parquet, csv, jsonl) or directly from
    Agent-lightning Dataset instances. Automatically splits into train/val if needed.

    Attributes:
        batch_size: Number of tasks per batch.
        n_epochs: Number of training epochs.
        train_file: Optional path to training data file.
        val_file: Optional path to validation data file.
        train_dataset: Optional in-memory training dataset.
        val_dataset: Optional in-memory validation dataset.
        train_val_split: Fraction of data to use for training.
        shuffle: Whether to shuffle the dataset.
        group_size: Number of rollouts per task group.
        seed: Random seed for shuffling.
    """

    batch_size: int
    n_epochs: int = 1
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
        """Build and return train and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset).

        Raises:
            ValueError: If no training dataset is provided.
        """
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
                n_epochs=self.n_epochs,
                shuffle=self.shuffle,
                group_size=self.group_size,
                seed=self.seed,
            ),
            # For validation, always use batch_size=len(val_dataset) and group_size=1 to avoid dropping or repeating any samples
            AGLDataset(val_dataset, batch_size=len(val_dataset), shuffle=False, group_size=1),
        )
