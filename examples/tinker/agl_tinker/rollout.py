# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Generic, List, Sequence, TypeVar, cast

from tinker.types import ModelInput
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.types import (
    Trajectory,
    TrajectoryGroup,
    Transition,
)

from agentlightning import LightningStore, Span, TraceToTripletBase
from agentlightning import Triplet as AGLTriplet
from agentlightning.types.core import RolloutMode

from .env import AGLDataset, AGLDummyEnv, AGLDummyEnvGroupBuilder

logger = logging.getLogger(__name__)

T_task = TypeVar("T_task")


def reconstruct_transitions(
    spans: List[Span], adapter: TraceToTripletBase, rollout_id: str, identical_credit_assignment: bool = True
) -> Trajectory:
    """Reconstruct the transitions from the spans.

    Args:
        spans: The spans to reconstruct the transitions from.
        adapter: The adapter to use to reconstruct the transitions.
        rollout_id: The ID of the rollout.
        identical_credit_assignment: Whether to assign the reward from later triplets to earlier triplets.
            See [Agent-lightning's paper](https://arxiv.org/abs/2508.03680) for details.

    Returns:
        Tinker trajectory.
    """
    triplets: List[AGLTriplet] = adapter.adapt(spans)
    # We need to reconstruct the input and output tokens (+logprobs) from the triplets
    transitions: list[Transition] = []

    # Initialize to a non-zero reward
    last_reward: float = 0.0

    for i_triplet, triplet in reversed(list(enumerate(triplets))):
        if "token_ids" not in triplet.prompt or "token_ids" not in triplet.response:
            logger.error(f"[Rollout {rollout_id}] Triplet has no token_ids: {triplet}")
            continue
        # Getting the input and output tokens from the triplet
        input_tokens = ModelInput.from_ints(triplet.prompt["token_ids"])
        output_tokens = triplet.response["token_ids"]
        # Logprobs sometimes are available too.
        if "logprobs" not in triplet.response:
            logger.warning(f"[Rollout {rollout_id}] Triplet has no logprobs: {triplet}")
            logprobs = None
        else:
            logprobs = [prob["logprob"] for prob in triplet.response["logprobs"]]
            if len(logprobs) != len(output_tokens):
                logger.warning(
                    f"[Rollout {rollout_id}] Triplet has {len(logprobs)} logprobs "
                    f"but {len(output_tokens)} output tokens: {triplet}"
                )
                logprobs = None
        output_tokens_with_logprobs = TokensWithLogprobs(tokens=output_tokens, maybe_logprobs=logprobs)

        if triplet.reward is None:
            if identical_credit_assignment:
                # takes from the next non-null reward
                this_reward = last_reward
            else:
                # Assume it to be zero
                this_reward = 0.0
        else:
            this_reward = triplet.reward

        transitions.append(
            Transition(
                ob=input_tokens,
                ac=output_tokens_with_logprobs,
                reward=this_reward,
                episode_done=i_triplet + 1 == len(triplets),
            )
        )

        if identical_credit_assignment and triplet.reward is not None:
            last_reward = triplet.reward

    # The final observation is empty input tokens
    return Trajectory(transitions=transitions[::-1], final_ob=ModelInput.from_ints([]))


async def agl_single_rollout(
    llm_resources_id: str, env: AGLDummyEnv[Any], store: LightningStore, adapter: TraceToTripletBase, mode: RolloutMode
) -> Trajectory:
    """Under Agent-lightning, there is no such thing as a "env".
    The "env" here is a simple wrapper around a task.
    """
    rollout = await store.enqueue_rollout(env.task, mode=mode, resources_id=llm_resources_id)

    while True:
        completed_rollout = await store.get_rollout_by_id(rollout.rollout_id)
        if completed_rollout is not None and completed_rollout.status in ["succeeded", "failed", "cancelled"]:
            break

        # Wait until the rollout is completed
        await asyncio.sleep(1.0)

    if completed_rollout.status != "succeeded":
        logger.error(f"[Rollout {rollout.rollout_id}] Failed with status {completed_rollout.status}")
    else:
        logger.info(
            f"[Rollout {rollout.rollout_id}] Rollout succeeded under "
            f"{cast(float, completed_rollout.end_time) - completed_rollout.start_time:.2f} seconds"
        )

    spans = await store.query_spans(rollout.rollout_id, "latest")
    if not spans:
        logger.error(f"[Rollout {rollout.rollout_id}] No spans found. Return an empty trajectory.")
        return Trajectory(transitions=[], final_ob=ModelInput.from_ints([]))

    triplets = adapter.adapt(spans)
    logger.info(
        f"[Rollout {rollout.rollout_id}] Adapted {len(triplets)} triplets from {len(spans)} spans. "
        f"Rewards are: {[t.reward for t in triplets]}"
    )

    # Converting triplets to Tinker transitions
    reconstructed = reconstruct_transitions(spans, adapter, rollout.rollout_id)
    logger.info(
        f"[Rollout {rollout.rollout_id}] Reconstructed {len(reconstructed.transitions)} transitions. "
        f"Rewards are: {[r.reward for r in reconstructed.transitions]}"
    )
    return reconstructed


async def agl_group_rollout(
    env_group_builder: AGLDummyEnvGroupBuilder[Any],
    llm_resources_id: str,
    store: LightningStore,
    adapter: TraceToTripletBase,
    mode: RolloutMode,
) -> TrajectoryGroup:
    envs_G: Sequence[AGLDummyEnv[Any]] = await env_group_builder.make_envs()
    trajectories_G = await asyncio.gather(
        *[agl_single_rollout(llm_resources_id, env, store, adapter, mode) for env in envs_G]
    )
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


def dataset_to_env_group_builders(dataset: AGLDataset[T_task]) -> list[AGLDummyEnvGroupBuilder[T_task]]:
    """
    Get the whole dataset as a list of env group builders.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


class AGLTestSetEvaluator(Generic[T_task]):
    """Run an evaluation on a test set."""

    def __init__(self, dataset: AGLDataset[T_task], name: str | None = None):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.name = name

    async def __call__(
        self, llm_resources_id: str, store: LightningStore, adapter: TraceToTripletBase, mode: RolloutMode
    ) -> dict[str, float]:
        trajectory_groups_P = await asyncio.gather(
            *[
                agl_group_rollout(builder, llm_resources_id, store, adapter, mode)
                for builder in self.env_group_builders_P
            ]
        )
        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)

        if self.name is not None:
            metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
