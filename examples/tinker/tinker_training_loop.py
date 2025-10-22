# Copyright (c) Microsoft. All rights reserved.

"""RL training main loop with Tinker API.

This script is based on Tinker Cookbook's [`rl_train.py` script](https://github.com/thinking-machines-lab/tinker-cookbook/blob/9b2af83cb62b9c4e8325a0efab71429e5aedf289/tinker_cookbook/rl/train.py),
with modifications on how the rollout is collected.

Environments are not used at all because Agent-lightning handles "environment" has part of user agent's logic.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from turtle import forward
from typing import Any, Callable, List, Literal, Sequence

import chz
import numpy as np
import tinker
import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import incorporate_kl_penalty
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    print_group,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.trace import get_scope_context, scope, trace_init

logger = logging.getLogger("agentlightning.tinker")


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    remove_constant_reward_groups: bool = False
    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None


@scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    do_remove_constant_reward_groups: bool,
) -> TrajectoryGroup | None:
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)
    trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Remove if all trajectories have the same reward
    trajectory_groups = [trajectory_group]
    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if len(trajectory_groups) == 0:
        return None
    return trajectory_groups[0]


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics: dict[str, Any] = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Print one trajectory
    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    context = get_scope_context()
    context.attributes["step"] = i_batch

    metrics: dict[str, Any] = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        service_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group for trajectory_group in trajectory_groups_P if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(cfg.model_name, rank=cfg.lora_rank)

    load_state_path: str | None = resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
