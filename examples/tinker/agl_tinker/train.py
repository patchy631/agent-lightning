# Copyright (c) Microsoft. All rights reserved.

"""RL training main loop with Tinker API.

This script is based on Tinker Cookbook's [`rl_train.py` script](https://github.com/thinking-machines-lab/tinker-cookbook/blob/9b2af83cb62b9c4e8325a0efab71429e5aedf289/tinker_cookbook/rl/train.py),
with modifications on how the rollout is collected.

Environments are not used at all because Agent-lightning handles "environment" has part of user agent's logic.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Literal, Sequence

import chz
import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.metrics import incorporate_kl_penalty
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    print_group,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.trace import get_scope_context, scope, trace_init

from agentlightning import (
    LightningStore,
    LightningStoreClient,
    LLMProxy,
    LlmProxyTraceToTriplet,
    TracerTraceToTriplet,
    TraceToTripletBase,
)

from .env import AGLDataset, AGLDatasetBuilder
from .llm import TinkerLLM
from .rollout import AGLTestSetEvaluator, do_group_of_group_rollouts

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: AGLDatasetBuilder[Any]  # also determines batch size
    model_name: str
    renderer_name: str
    compute_post_kl: bool = False
    lora_rank: int = 32
    llm_proxy_port: int = 12306

    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Sampling parameters
    max_tokens: int = 2048
    train_temperature: float = 1.0
    eval_temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0

    # Agent-lightning parameters (only used in when running standalone)
    store_address: str = "http://localhost:4747"
    # Using tracing data from LLM proxy instead of client-side tracer
    # When this is true, adapter_agent_match is ignored
    adapter_from_llm_proxy: bool = False
    adapter_agent_match: str | None = None
    llm_proxy_retry_attempts: int = 0

    # Concurrency parameters (mainly for controlling max queue length)
    concurrency: int = 16

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
    *,
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[AGLTestSetEvaluator[Any]],
    dataset: AGLDataset[Any],
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    store: LightningStore,
    adapter: TraceToTripletBase,
    llm_proxy: LLMProxy,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    logger.info(f"Creating sampling client with training client {training_client} and start batch {start_batch}")
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )
    logger.info(f"Creating renderer with name {cfg.renderer_name}")
    renderer = get_renderer(cfg.renderer_name, tokenizer)

    tinker_llm = TinkerLLM(
        model_name=cfg.model_name,
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        max_tokens=cfg.max_tokens,
        temperature=cfg.train_temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
    ).rewrite_litellm_custom_providers()

    logger.info(f"Starting training from batch {start_batch} to {end_batch}")
    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        logger.info(f"[Batch {i_batch}] Starting training step. Learning rate: {cfg.learning_rate}")
        t_start = time.time()

        llm_proxy.update_model_list(tinker_llm.as_model_list())
        llm_proxy.restart()

        logger.info(f"[Batch {i_batch}] LiteLLM model list: {llm_proxy.model_list}")
        llm_resource = llm_proxy.as_resource()
        resources_update = await store.add_resources({"main_llm": llm_resource})

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            logger.info(f"[Batch {i_batch}] Running evaluations")
            tinker_llm.temperature = cfg.eval_temperature
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(resources_update.resources_id, store, adapter, "val", i_batch)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})
            tinker_llm.temperature = cfg.train_temperature

        # Get batch and sample trajectories
        logger.info(f"[Batch {i_batch}] Getting batch data from dataset")
        env_group_builders_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            logger.info(f"[Batch {i_batch}] Sampling trajectories...")
            trajectory_groups_P = await do_group_of_group_rollouts(
                env_group_builders_P,
                resources_update.resources_id,
                i_batch,
                store=store,
                adapter=adapter,
                mode="train",
                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                concurrency=cfg.concurrency,
            )
        logger.info(f"[Batch {i_batch}] Trajectories sampled: {len(trajectory_groups_P)}")

        # Train step
        logger.info(f"[Batch {i_batch}] Starting training step...")
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
        )
        # Point Tinker LLM to a new model
        tinker_llm.update_sampling_client(sampling_client)

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)
        logger.info(f"[Batch {i_batch}] Sampling and training completed")

    llm_proxy.stop()


@scope
async def main_training_loop(
    cfg: Config,
    store: LightningStore,
    adapter: TraceToTripletBase,
    llm_proxy: LLMProxy,
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
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    logger.info(f"Creating service client with base URL {cfg.base_url}")
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    logger.info(f"Creating training client with model name {cfg.model_name} and rank {cfg.lora_rank}")
    training_client = await service_client.create_lora_training_client_async(cfg.model_name, rank=cfg.lora_rank)

    load_state_path: str | None = resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")
    else:
        logger.info("No checkpoint found, starting from scratch")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()
    logger.info(f"Tokenizer created: {tokenizer}")

    # Create dataset from thunk
    dataset, test_dataset = await cfg.dataset_builder()
    evaluators = [AGLTestSetEvaluator(test_dataset)]

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches and test on {len(test_dataset)} batches")

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
        store=store,
        adapter=adapter,
        llm_proxy=llm_proxy,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        logger.info(f"Saving final checkpoint to {cfg.log_path}/final.pt")
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


@scope
async def main(config: Config) -> None:
    store = LightningStoreClient(config.store_address)
    if config.adapter_from_llm_proxy:
        # This is still under development
        if config.adapter_agent_match is not None:
            raise ValueError("adapter_agent_match is not supported when adapter_from_llm_proxy is True")
        adapter = LlmProxyTraceToTriplet()
    else:
        # This is the tested path
        adapter = TracerTraceToTriplet(agent_match=config.adapter_agent_match, _skip_empty_token_spans=True)
    llm_proxy = LLMProxy(
        port=config.llm_proxy_port,
        model_list=[],
        store=store,
        num_retries=config.llm_proxy_retry_attempts,
    )

    await main_training_loop(config, store, adapter, llm_proxy)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
