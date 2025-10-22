from __future__ import annotations

from random import Random
from typing import Generic, Sequence, TypeVar

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    StepResult,
    StopCondition,
    Trajectory,
)

from agentlightning import Dataset

T_task = TypeVar("T_task")


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

    async def make_envs(self) -> Sequence[Env]:
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
