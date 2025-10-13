# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to use an existing APO algorithm to tune the prompts."""

from typing import List, Optional

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import Trainer, configure_logger
from agentlightning.algorithm.base import algo
from agentlightning.litagent.decorator import rollout
from agentlightning.reward import find_final_reward
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, PromptTemplate, Span

console = Console()


@rollout
async def apo_rollout(task: str, prompt_template: PromptTemplate) -> float:
    # This relies on a public OpenAI service
    client = AsyncOpenAI()

    result = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": prompt_template.format(any_question=task)},
        ],
    )

    text = result.choices[0].message.content
    console.print(f"[bold yellow][Rollout][/bold yellow] LLM returned: {text}")

    return await llm_judge(task, text)


async def llm_judge(task: str, output: Optional[str]) -> float:
    client = AsyncOpenAI()
    judge_prompt = f"""Evaluate how well the output fulfills the task.
Task: {task}
Output: {output}
You must be very critical and strict in your evaluation.
Return only a number between 0 and 1. No text, punctuation, or explanation."""
    result = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0.0,
    )
    try:
        content = result.choices[0].message.content
        if content is None:
            console.print(f"[bold blue][Judge][/bold blue] Judge retured no content: {result}")
            return 0.0
        score = float(content)
        console.print(f"[bold blue][Judge][/bold blue] Judge returned score: {score}")
        return score
    except ValueError:
        console.print(f"[bold blue][Judge][/bold blue] Error evaluating output: {result}")
        return 0.0


if __name__ == "__main__":
    configure_logger()
    trainer = Trainer(n_workers=1, algorithm=apo_algorithm)
    trainer.fit_v2(apo_rollout)
