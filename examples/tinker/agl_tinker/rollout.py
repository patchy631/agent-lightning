# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Sequence, cast

from tinker.types import ModelInput
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import Message, Qwen3Renderer
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from transformers import AutoTokenizer

from agentlightning import LLM, LightningStore, TraceToTripletBase

from .env import AGLDummyEnv, AGLDummyEnvGroupBuilder

logger = logging.getLogger(__name__)


async def agl_single_rollout(
    llm: LLM, env: AGLDummyEnv[Any], store: LightningStore, adapter: TraceToTripletBase
) -> Trajectory:
    """Under Agent-lightning, there is no such thing as a "env".
    The "env" here is a simple wrapper around a task.
    """
    rollout = await store.enqueue_rollout(env.task)

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
            f"[Rollout {rollout.rollout_id}] Succeeded under {cast(float, completed_rollout.end_time) - completed_rollout.start_time:.2f} seconds"
        )

    spans = await store.query_spans(rollout.rollout_id, "latest")
    triplets = adapter.adapt(spans)
    logger.info(
        f"[Rollout {rollout.rollout_id}] Adapted {len(triplets)} triplets from {len(spans)} spans. "
        f"Rewards are: {[t.reward for t in triplets]}"
    )

    # Converting triplets to Tinker transitions
    # We need to reconstruct the input and output tokens (+logprobs) from the triplets
    transitions: list[Transition] = []
    for i_triplet, triplet in enumerate(triplets):
        if "token_ids" not in triplet.prompt or "token_ids" not in triplet.response:
            logger.error(f"[Rollout {rollout.rollout_id}] Triplet has no token_ids: {triplet}")
            continue
        # Getting the input and output tokens from the triplet
        input_tokens = ModelInput.from_ints(triplet.prompt["token_ids"])
        output_tokens = triplet.response["token_ids"]
        # Logprobs sometimes are available too.
        if "logprobs" not in triplet.response:
            logger.warning(f"[Rollout {rollout.rollout_id}] Triplet has no logprobs: {triplet}")
            logprobs = None
        else:
            logprobs = triplet.response["logprobs"]
        output_tokens_with_logprobs = TokensWithLogprobs(tokens=output_tokens, maybe_logprobs=logprobs)
        transitions.append(
            Transition(
                ob=input_tokens,
                ac=output_tokens_with_logprobs,
                reward=triplet.reward if triplet.reward is not None else 0.0,
                episode_done=i_triplet + 1 == len(triplets),
            )
        )

    # The final observation is empty input tokens
    return Trajectory(transitions=transitions, final_ob=ModelInput.from_ints([]))


async def agl_group_rollout(
    env_group_builder: AGLDummyEnvGroupBuilder[Any], llm: LLM, store: LightningStore, adapter: TraceToTripletBase
) -> TrajectoryGroup:
    envs_G: Sequence[AGLDummyEnv[Any]] = await env_group_builder.make_envs()
    trajectories_G = await asyncio.gather(*[agl_single_rollout(llm, env, store, adapter) for env in envs_G])
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
