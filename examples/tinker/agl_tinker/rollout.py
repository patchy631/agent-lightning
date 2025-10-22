from __future__ import annotations

import asyncio
import random
from typing import Any, Sequence

from tinker.types import ModelInput
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.renderers import Qwen3Renderer
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .env import AGLDummyEnv, AGLDummyEnvGroupBuilder

renderer = Qwen3Renderer(AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507"))

from tinker_cookbook.renderers import Message


async def agl_single_rollout(policy: TokenCompleter, env: AGLDummyEnv[Any]) -> Trajectory:
    transitions: list[Transition] = []
    input_message = str(env.task)
    print("Input message:", input_message)
    model_input = renderer.build_generation_prompt([Message(role="user", content=input_message)])
    model_output = await policy(model_input, renderer.get_stop_sequences())
    print("Model output:", model_output)

    transitions.append(
        Transition(ob=model_input, ac=model_output, reward=random.uniform(0, 1), episode_done=True, metrics={})
    )

    return Trajectory(transitions=transitions, final_ob=ModelInput.from_ints([]))


async def agl_group_rollout(env_group_builder: AGLDummyEnvGroupBuilder[Any], policy: TokenCompleter) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    trajectories_G = await asyncio.gather(*[agl_single_rollout(policy, env) for env in envs_G])
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
