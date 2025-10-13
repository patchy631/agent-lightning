# Copyright (c) Microsoft. All rights reserved.

"""
APO with textual gradients that read rollout spans and outputs to modify the prompt.

- algo: beam search with span-aware textual gradients -> apply_edit via LLM
- rollout: same pattern as your example, but task is a dict (T_task)

Based on the idea of:

- ProTeGi: https://aclanthology.org/2023.emnlp-main.494.pdf
- TextGrad: https://github.com/zou-group/textgrad
"""

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypedDict, TypeVar, cast

import poml
from openai import AsyncOpenAI

from agentlightning.adapter.messages import TraceMessagesAdapter
from agentlightning.algorithm.base import BaseAlgorithm, algo
from agentlightning.reward import find_final_reward
from agentlightning.types import Dataset, NamedResources, PromptTemplate, RolloutMode, RolloutStatus, RolloutV2, Span

logger = logging.getLogger(__name__)

T_task = TypeVar("T_task", bound=Dict[str, Any])


class RolloutResultForAPO(TypedDict):
    """This must be all JSON serializable to be processable by POML."""

    status: RolloutStatus
    final_reward: Optional[float]
    spans: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]


GRADIENT_PROMPT_FILES = [
    Path(__file__).parent / "prompts" / "text_gradient_variant01.poml",
    Path(__file__).parent / "prompts" / "text_gradient_variant02.poml",
]

APPLY_EDIT_PROMPT_FILES = [
    Path(__file__).parent / "prompts" / "apply_edit_variant01.poml",
    Path(__file__).parent / "prompts" / "apply_edit_variant02.poml",
]


def batch_iter_over_dataset(dataset: Dataset[T_task], batch_size: int) -> Iterator[Sequence[T_task]]:
    if batch_size <= len(dataset):
        while True:
            dataset_copy = [dataset[i] for i in range(len(dataset))]
            random.shuffle(dataset_copy)
            yield dataset_copy

    else:
        current_batch: List[int] = []
        while True:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            for index in indices:
                if index in current_batch:
                    continue
                current_batch.append(index)
                if len(current_batch) == batch_size:
                    yield [dataset[index] for index in current_batch]
                    current_batch = []


class APO(BaseAlgorithm, Generic[T_task]):

    def __init__(
        self,
        async_openai_client: AsyncOpenAI,
        *,
        gradient_model: str = "gpt-5-mini",
        apply_edit_model: str = "gpt-4.1-mini",
        diversity_temperature: float = 1.0,
        rollout_batch_size: int = 16,
        gradient_batch_size: int = 4,
        val_batch_size: int = 16,
        beam_width: int = 4,
        search_width: int = 4,
        beam_rounds: int = 3,
        rollout_batch_timeout: float = 600.0,
    ):
        self.async_openai_client = async_openai_client
        self.gradient_model = gradient_model
        self.apply_edit_model = apply_edit_model
        self.diversity_temperature = diversity_temperature
        self.gradient_batch_size = gradient_batch_size
        self.val_batch_size = val_batch_size
        self.beam_width = beam_width
        self.search_width = search_width
        self.beam_rounds = beam_rounds
        self.rollout_batch_timeout = rollout_batch_timeout

        self._history_best_prompt: Optional[PromptTemplate] = None
        self._history_best_score: float = float("-inf")

    def get_seed_prompt_template(self) -> Tuple[str, PromptTemplate]:
        initial_resources = self.get_initial_resources()
        if initial_resources is None:
            raise ValueError(
                "initial_resources are not set for APO algorithm. "
                "Use algorithm.set_initial_resources() to set initial resources or set it in Trainer()"
            )
        for name, resource in initial_resources.items():
            if isinstance(resource, PromptTemplate):
                return name, resource
        raise ValueError("No prompt template resource found in initial_resources")

    def get_adapter(self) -> TraceMessagesAdapter:
        adapter = super().get_adapter()
        if not isinstance(adapter, TraceMessagesAdapter):
            raise ValueError("Adapter must be a TraceMessagesAdapter for APO algorithm")
        return adapter

    def get_best_prompt(self) -> PromptTemplate:
        if self._history_best_prompt is None:
            raise ValueError("No best prompt found")
        return self._history_best_prompt

    async def _compute_textual_gradient(
        self,
        current_prompt: str,
        rollout_results: List[RolloutResultForAPO],
    ) -> Optional[str]:
        """
        Compute critique from spans + outputs.
        """
        tg_template = random.choice(GRADIENT_PROMPT_FILES)

        if len(rollout_results) < self.gradient_batch_size:
            logger.warning(
                f"Only {len(rollout_results)} rollouts available, but {self.gradient_batch_size} are needed. Using all rollouts."
            )
            sampled_rollout_results = rollout_results
        else:
            sampled_rollout_results = random.sample(rollout_results, self.gradient_batch_size)

        logger.info(
            f"Gradient will be computed with {self.gradient_model} for {len(sampled_rollout_results)} rollouts with template: {tg_template}"
        )

        tg_msg = poml.poml(  # type: ignore
            tg_template,
            context={
                "experiments": sampled_rollout_results,
                "prompt_template": current_prompt,
            },
            format="openai_chat",
        )
        logger.debug(f"Gradient computed with{self.gradient_model} prompt: {tg_msg}")
        critique_response = await self.async_openai_client.chat.completions.create(
            model=self.gradient_model,
            messages=tg_msg["messages"],  # type: ignore
            temperature=self.diversity_temperature,
        )
        critique_text = critique_response.choices[0].message.content
        logger.debug(f"Gradient computed with {self.gradient_model} has result: {critique_text}")

        return critique_text

    async def textual_gradient_and_apply_edit(
        self,
        current_prompt: str,
        rollout: List[RolloutResultForAPO],
    ) -> Optional[str]:
        """
        Compute critique from spans + outputs, then rewrite the prompt.
        """
        # 1) Critique
        critique_text = await self._compute_textual_gradient(current_prompt, rollout)
        if not critique_text:
            logger.error(f"Failed to compute critique for prompt.")
            return current_prompt

        # 2) Apply edit
        ae_template = random.choice(APPLY_EDIT_PROMPT_FILES)
        logger.info(f"Edit will be generated by {self.apply_edit_model} with template: {ae_template}")
        ae_msg = poml.poml(  # type: ignore
            ae_template,
            context={
                "prompt_template": current_prompt,
                "critique": critique_text,
            },
            format="openai_chat",
        )

        ae_response = await self.async_openai_client.chat.completions.create(
            model=self.apply_edit_model,
            messages=ae_msg["messages"],  # type: ignore
            temperature=self.diversity_temperature,
        )
        new_prompt = ae_response.choices[0].message.content
        if new_prompt:
            logger.info(f"Edit generated by {self.apply_edit_model}: {new_prompt}")
        return new_prompt

    async def get_rollout_results(self, rollout: List[RolloutV2]) -> List[RolloutResultForAPO]:
        rollout_results: List[RolloutResultForAPO] = []
        store = self.get_store()
        adapter = self.get_adapter()
        for r in rollout:
            spans = await store.query_spans(r.rollout_id)
            messages = adapter.adapt(spans)
            rollout_result = RolloutResultForAPO(
                status=r.status,
                final_reward=find_final_reward(spans),
                spans=[span.model_dump() for span in spans],
                messages=[m.model_dump() for m in messages],
            )
            logger.info(
                f"Rollout result for {r.rollout_id}: status {rollout_result['status']} "
                f"with final reward {rollout_result['final_reward']}. "
                f"{len(rollout_result['spans'])} spans and {len(rollout_result['messages'])} messages."
            )
            rollout_results.append(rollout_result)
        return rollout_results

    async def evaluate_prompt_on_batch(
        self,
        prompt: str,
        resource_name: str,
        dataset: Sequence[T_task],
        mode: RolloutMode,
    ) -> Tuple[List[RolloutResultForAPO], float]:
        """
        Enqueue one rollout per example. Aggregate mean reward.
        Return average reward and per-example spans+reward for gradient steps.
        """
        store = self.get_store()
        logger.info(f'Evaluating prompt "{prompt[:50]}..." on {len(dataset)} tasks in {mode} mode')

        # Install prompt as named resource
        resources: NamedResources = {resource_name: PromptTemplate(template=prompt, engine="f-string")}
        await store.add_resources(resources)

        rollout_ids: List[str] = []
        for t in dataset:
            r = await store.enqueue_rollout(input=t, mode=mode)  # task can be any dict processable by the client
            rollout_ids.append(r.rollout_id)

        deadline = time.time() + self.rollout_batch_timeout
        finished: List[RolloutV2] = []
        while time.time() < deadline:
            finished = await store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=0.0)
            if len(finished) >= len(rollout_ids):
                logger.info(f"All {len(rollout_ids)} rollouts finished within timeout.")
                break

        rollout_results = await self.get_rollout_results(finished)
        final_rewards = [rr["final_reward"] for rr in rollout_results]

        avg = float(sum([r or 0.0 for r in final_rewards]) / max(1, len(final_rewards)))

        logger.info(f"Evaluated {len(rollout_results)} rollouts. Rewards: {final_rewards}. Average reward: {avg}")
        return rollout_results, avg

    async def run(
        self,
        train_dataset: Optional[Dataset[T_task]] = None,
        val_dataset: Optional[Dataset[T_task]] = None,
    ) -> None:
        # Get the initial prompt template
        resource_name, seed_prompt = self.get_seed_prompt_template()

        if train_dataset is None:
            raise ValueError("train_dataset is required for APO algorithm")
        if val_dataset is None:
            raise ValueError("val_dataset is required for APO algorithm")

        grad_dataset_iterator = batch_iter_over_dataset(train_dataset, self.gradient_batch_size)
        val_dataset_iterator = batch_iter_over_dataset(val_dataset, self.val_batch_size)

        self._history_best_prompt = seed_prompt
        self._history_best_score = float("-inf")

        beam: List[PromptTemplate] = [seed_prompt]
        for rnd in range(self.beam_rounds):
            logger.info(f"[Round {rnd + 1}] Round {rnd + 1}/{self.beam_rounds}...")
            if len(beam) < self.beam_width:
                logger.warning(
                    f"[Round {rnd + 1}] Beam width is currently {self.beam_width}, but only {len(beam)} prompts in beam. "
                    "Replicating all prompts."
                )
                parent_prompts = [beam[i % len(beam)] for i in range(self.beam_width)]
            else:
                parent_prompts = random.sample(beam, self.beam_width)

            next_beam: List[PromptTemplate] = [*beam]

            # When computing gradienst, use a different subset data for each iteration
            logger.info(
                f"[Round {rnd + 1}] Applying {self.search_width} edits to each of "
                f"the {len(parent_prompts)} parents on training dataset"
            )
            for prompt in parent_prompts:
                for _ in range(self.search_width):
                    grad_samples = next(grad_dataset_iterator)
                    rollout_results, _ = await self.evaluate_prompt_on_batch(
                        prompt.template, resource_name, grad_samples, mode="train"
                    )
                    new_prompt = await self.textual_gradient_and_apply_edit(prompt.template, rollout_results)
                    if not new_prompt:
                        logger.error(f"[Round {rnd + 1}] Failed to compute edit for prompt: {prompt.template}")
                        continue
                    new_prompt_template = PromptTemplate(template=new_prompt, engine="f-string")
                    logger.info(f"[Round {rnd + 1}] New prompt template: {new_prompt_template}")
                    next_beam.append(new_prompt_template)

            # Evaluate on candidates on VAL
            logger.info(f"[Round {rnd + 1}] Evaluating {len(next_beam)} candidates on validation dataset")
            val_batch = next(val_dataset_iterator)
            scores: List[Tuple[PromptTemplate, float]] = []
            for idx, prompt in enumerate(next_beam):
                rollout_results, score = await self.evaluate_prompt_on_batch(
                    prompt.template, resource_name, val_batch, mode="val"
                )
                scores.append((prompt, score))
                logger.info(f"[Round {rnd + 1}] Candidate {idx} score: {score:.3f}")
            # Sort the beam by score
            next_beam = [p for p, _ in sorted(scores, key=lambda x: x[1], reverse=True)][: self.beam_width]
            logger.info(f"[Round {rnd + 1}] Top {len(next_beam)} candidates on validation dataset: {next_beam}")

            # Add the best prompt to the history
            if len(next_beam) == 0:
                raise ValueError("No beam candidates any more")
            best_prompt = next_beam[0]
            _, best_score = await self.evaluate_prompt_on_batch(
                best_prompt.template, resource_name, cast(Sequence[T_task], val_dataset), mode="val"
            )
            logger.info(f"[Round {rnd + 1}] Best prompt {best_prompt} has score: {best_score:.3f}")

            if best_score > self._history_best_score:
                logger.info(
                    f"[Round {rnd + 1}] Best prompt updated. New best score: {best_score:.3f} (prev: {self._history_best_score:.3f})"
                )
                self._history_best_prompt = best_prompt
                self._history_best_score = best_score
