# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from random import Random
from typing import Generic, Optional, Sequence, TypeVar

import chz
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    StopCondition,
    Trajectory,
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

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
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

    train_dataset: Dataset[T_task]
    batch_size: int
    val_dataset: Optional[Dataset[T_task]] = None
    train_val_split: float = 0.7
    shuffle: bool = True
    group_size: int = 4
    seed: int = 42

    async def __call__(self) -> tuple[AGLDataset[T_task], AGLDataset[T_task]]:
        if self.val_dataset is None:
            indices = list(range(len(self.train_dataset)))
            Random(self.seed).shuffle(indices)
            val_indices = indices[int(len(indices) * self.train_val_split) :]
            train_indices = indices[: int(len(indices) * self.train_val_split)]
            logger.warning(
                "No validation dataset provided, splitting train dataset into train (%d) and validation (%d)",
                len(train_indices),
                len(val_indices),
            )
            train_dataset = [self.train_dataset[i] for i in train_indices]
            val_dataset = [self.train_dataset[i] for i in val_indices]
        else:
            train_dataset = self.train_dataset
            val_dataset = self.val_dataset

        return (
            AGLDataset(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                group_size=self.group_size,
                seed=self.seed,
            ),
            # For validation, always use batch_size=1 and group_size=1 to avoid dropping or repeating any samples
            AGLDataset(val_dataset, batch_size=1, shuffle=False, group_size=1),
        )


class AGLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(self, dataset: RLDataset, max_tokens: int, name: str | None = None):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.max_tokens = max_tokens
        self.name = name

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        trajectory_groups_P = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in self.env_group_builders_P]
        )
        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)

        if self.name is not None:
            metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
