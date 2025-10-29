# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import itertools
import logging
import random
from typing import Any, Dict, Generic, List, Sequence, Tuple, TypeVar, cast

from tinker.types import ModelInput
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import remove_constant_reward_groups
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.types import (
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.utils.trace import scope

from agentlightning import LightningStore, Rollout, RolloutMode, RolloutStatus, Span, TraceToTripletBase
from agentlightning import Triplet as AGLTriplet

from .env import AGLDataset, AGLDummyEnv, AGLDummyEnvGroupBuilder

logger = logging.getLogger(__name__)

T_task = TypeVar("T_task")

WAIT_FOR_ROLLOUTS_INTERVAL = 5.0


def reconstruct_transitions(spans: List[Span], adapter: TraceToTripletBase, rollout_id: str) -> Trajectory:
    """Reconstruct the transitions from the spans.

    Args:
        spans: The spans to reconstruct the transitions from.
        adapter: The adapter to use to reconstruct the transitions.
        rollout_id: The ID of the rollout.

    Returns:
        Tinker trajectory.
    """
    triplets: List[AGLTriplet] = adapter.adapt(spans)
    # We need to reconstruct the input and output tokens (+logprobs) from the triplets
    transitions: list[Transition] = []

    for i_triplet, triplet in reversed(list(enumerate(triplets))):
        if "token_ids" not in triplet.prompt or "token_ids" not in triplet.response:
            logger.error(f"[Rollout {rollout_id}] Triplet has no token_ids: {triplet}. Skipping.")
            continue
        # TODO: Sometimes triplet.prompt is an empty list. This might be a bug with the adapter.
        if not triplet.prompt["token_ids"] or not triplet.response["token_ids"]:
            logger.warning(f"[Rollout {rollout_id}] Triplet has empty token_ids: {triplet}. Skipping.")
            continue
        # Getting the input and output tokens from the triplet
        input_tokens = ModelInput.from_ints(triplet.prompt["token_ids"])
        output_tokens = triplet.response["token_ids"]
        # Logprobs sometimes are available too.
        if "logprobs" not in triplet.response:
            logger.error(f"[Rollout {rollout_id}] Triplet has token_ids but no logprobs: {triplet}")
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

        # Log extra metrics for the reward in final step
        metrics: Dict[str, float] = {}
        if triplet.reward is not None and i_triplet + 1 == len(triplets):
            metrics["reward/final_step"] = triplet.reward
        metrics["reward/not_null"] = 1.0 if triplet.reward is not None else 0.0

        # TODO: The logic below might cause failed rollouts to be treated as rollouts with 0 reward.
        # We log that at the moment, but we should consider a better way to handle this.
        transitions.append(
            Transition(
                ob=input_tokens,
                ac=output_tokens_with_logprobs,
                # For no-reward, we fill it with 0.0.
                # Later, this step is not taken into trajectory-level advantage calculation.
                reward=triplet.reward if triplet.reward is not None else 0.0,
                episode_done=i_triplet + 1 == len(triplets),
                metrics=metrics,
            )
        )

    # The final observation is empty input tokens
    return Trajectory(transitions=transitions[::-1], final_ob=ModelInput.from_ints([]))


async def agl_single_rollout(
    llm_resources_id: str,
    env: AGLDummyEnv[Any],
    *,
    store: LightningStore,
    adapter: TraceToTripletBase,
    mode: RolloutMode,
) -> Tuple[Rollout, Trajectory]:
    """Under Agent-lightning, there is no such thing as a "env".
    The "env" here is a simple wrapper around a task.
    """
    rollout_partial = await store.enqueue_rollout(env.task, mode=mode, resources_id=llm_resources_id)

    while True:
        completed_rollout = await store.get_rollout_by_id(rollout_partial.rollout_id)
        if completed_rollout is not None and completed_rollout.status in ["succeeded", "failed", "cancelled"]:
            break

        # Wait until the rollout is completed
        # This should be a slightly large number to avoid busy-waiting.
        # Add a small jitter to avoid synchronized waiting.
        jitter = random.uniform(0.9, 1.1)
        await asyncio.sleep(WAIT_FOR_ROLLOUTS_INTERVAL * jitter)

    if completed_rollout.status != "succeeded":
        logger.error(f"[Rollout {completed_rollout.rollout_id}] Failed with status {completed_rollout.status}")
    else:
        logger.debug(
            f"[Rollout {completed_rollout.rollout_id}] Rollout succeeded under "
            f"{cast(float, completed_rollout.end_time) - completed_rollout.start_time:.2f} seconds"
        )

    spans = await store.query_spans(completed_rollout.rollout_id, "latest")
    if not spans:
        logger.error(f"[Rollout {completed_rollout.rollout_id}] No spans found. Return an empty trajectory.")
        return completed_rollout, Trajectory(transitions=[], final_ob=ModelInput.from_ints([]))

    triplets = adapter.adapt(spans)
    logger.debug(
        f"[Rollout {completed_rollout.rollout_id}] Adapted {len(triplets)} triplets from {len(spans)} spans. "
        f"Rewards are: {[t.reward for t in triplets]}"
    )

    # Converting triplets to Tinker transitions
    # Always do this no matter the rollout status is succeeded or not.
    reconstructed = reconstruct_transitions(spans, adapter, completed_rollout.rollout_id)
    logger.info(
        f"[Rollout {completed_rollout.rollout_id}] Reconstructed {len(reconstructed.transitions)} transitions from {len(spans)} spans. "
        f"Rewards are: {[r.reward for r in reconstructed.transitions]} (raw triplets rewards: {[t.reward for t in triplets]})"
    )
    return completed_rollout, reconstructed


@scope
async def do_group_of_group_rollouts(
    env_group_builders_P: Sequence[AGLDummyEnvGroupBuilder[Any]],
    llm_resources_id: str,
    i_batch: int,
    *,
    store: LightningStore,
    adapter: TraceToTripletBase,
    mode: RolloutMode,
    do_remove_constant_reward_groups: bool = False,
    concurrency: int = 16,
) -> List[TrajectoryGroup]:
    """
    Run rollouts for multiple env groups with a global concurrency limit that controls
    how many agl_single_rollout tasks may run at once (across *all* groups).

    Returns a list of TrajectoryGroup objects (one per env_group_builder), optionally
    filtered to remove groups whose trajectories all have the same reward.
    """
    # 1) Build all envs upfront (does not consume concurrency).
    groups_envs: List[Sequence[AGLDummyEnv[Any]]] = []
    for i, builder in enumerate(env_group_builders_P):
        envs = await builder.make_envs()
        if not envs:
            logger.warning(
                f"[Batch {i_batch} {mode}] [Group {i}] Builder produced no envs; "
                "returning empty group after compute step."
            )
        groups_envs.append(envs)

    # 2) Create a global semaphore to cap concurrent single rollouts.
    sem = asyncio.Semaphore(concurrency)

    # 3) For each env in each group, prepare a task that respects the semaphore.
    async def run_single_with_limit(env: AGLDummyEnv[Any]) -> Tuple[Rollout, Trajectory]:
        async with sem:
            return await agl_single_rollout(
                llm_resources_id,
                env,
                store=store,
                adapter=adapter,
                mode=mode,
            )

    # We keep tasks organized per group so we can compute group rewards afterward.
    per_group_tasks: List[List[asyncio.Task[Tuple[Rollout, Trajectory]]]] = []
    for group_idx, envs in enumerate(groups_envs):
        tasks = [asyncio.create_task(run_single_with_limit(env)) for env in envs]
        per_group_tasks.append(tasks)

    # 4) Await all groups, but still allow interleaving via the shared semaphore.
    trajectory_groups: List[TrajectoryGroup] = []
    for group_idx, (builder, tasks) in enumerate(zip(env_group_builders_P, per_group_tasks)):
        rollouts_and_trajectories_G = await asyncio.gather(*tasks)
        rollouts_G, trajectories_G = cast(
            Tuple[List[Rollout], List[Trajectory]], zip(*rollouts_and_trajectories_G, strict=True)
        )
        # Compute rewards/metrics for this group.
        rewards_and_metrics_G = await builder.compute_group_rewards(trajectories_G)
        rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

        # Attach AGL-specific metrics for error handling.
        metrics_agl: Dict[str, float | int] = {}
        metrics_agl["by_group/frac_empty_traj"] = sum(1 for traj in trajectories_G if not traj.transitions) / len(
            trajectories_G
        )
        completed_statuses: List[RolloutStatus] = ["succeeded", "failed", "cancelled"]
        for status in completed_statuses:
            metrics_agl[f"by_group/frac_status_{status}"] = sum(
                1 if rollout.status == status else 0 for rollout in rollouts_G
            ) / len(trajectories_G)
        metrics_agl["by_group/frac_status_others"] = sum(
            1 if rollout.status not in completed_statuses else 0 for rollout in rollouts_G
        ) / len(trajectories_G)

        tg = TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G) + [metrics_agl])
        trajectory_groups.append(tg)
        logger.info(
            f"[Batch {i_batch} {mode}] [Group {group_idx}] Completed {len(trajectories_G)} trajectories; "
            f"rewards: {[[trans.reward for trans in traj.transitions] for traj in trajectories_G]}"
        )

    # 5) Optional filtering of constant-reward groups (same behavior as before).
    if do_remove_constant_reward_groups:
        before = len(trajectory_groups)
        trajectory_groups = remove_constant_reward_groups(trajectory_groups)
        after = len(trajectory_groups)
        logger.info(
            f"[Batch {i_batch} {mode}] [Filter] Removed {before - after} constant-reward group(s); {after} remaining."
        )

    return trajectory_groups


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
        self, llm_resources_id: str, store: LightningStore, adapter: TraceToTripletBase, mode: RolloutMode, i_batch: int
    ) -> dict[str, float]:
        trajectory_groups_P = await do_group_of_group_rollouts(
            self.env_group_builders_P,
            llm_resources_id,
            i_batch=i_batch,
            store=store,
            adapter=adapter,
            mode=mode,
            do_remove_constant_reward_groups=False,
        )
        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)

        if self.name is not None:
            metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
