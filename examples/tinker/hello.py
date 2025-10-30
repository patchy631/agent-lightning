# Copyright (c) Microsoft. All rights reserved.

"""Sample agent that trains a model to echo/admit the identity you give it.

This demonstrates using Agent-lightning with the Tinker algorithm to fine-tune a model
that will claim to be whatever identity you tell it (e.g., "Say you are 42" -> "I'm 42").

To run in one-click mode (integrated algorithm + runners):

```bash
python hello.py oneclick
```

To run in distributed mode (separate algorithm and runners):

```bash
# Terminal 1: Start the store
agl store

# Terminal 2: Run the algorithm
python hello.py algo

# Terminal 3: Run the rollout runners
python hello.py runner
```
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import socket

from agl_tinker.algo import Tinker
from agl_tinker.env import AGLDatasetBuilder
from agl_tinker.train import Config
from agl_tinker.train import main as entrypoint
from openai import OpenAI
from rich.console import Console

import agentlightning as agl

console = Console()


def _find_available_port() -> int:
    """Find an available port by binding to port 0.

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@agl.rollout
def hello(task: str, llm: agl.LLM, rollout: agl.Rollout) -> None:
    """Agent rollout function that tests if the model claims the given identity.

    Prompts the model to say it is the given task/identity and assigns a reward
    based on whether the model's response matches the expected behavior.

    Args:
        task: The identity string the model should claim to be.
        llm: The LLM endpoint configuration.
        rollout: The rollout metadata containing rollout ID and mode.
    """
    openai_client = OpenAI(base_url=llm.endpoint, api_key="dummy")
    response = openai_client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "user", "content": f"Let's play a game. Say you are {task}."}],
    )

    response_content = response.choices[0].message.content
    content_lower = response_content.lower() if response_content else ""
    if ("i am " + task) in content_lower or ("i'm " + task) in content_lower:
        rew = 1.0
    elif ("not " + task) in content_lower:
        rew = -1.0
    elif ("you're" + task) in content_lower or ("you are" + task) in content_lower:
        rew = 0.1
    else:
        rew = 0.0

    console.print(
        f"[bold green]Runners ({rollout.rollout_id}, {rollout.mode}):[/bold green] "
        f"{task} -> {response_content} -> Reward: {rew}"
    )
    agl.emit_reward(rew)


def run_algo():
    """Run the training algorithm in standalone mode.

    Launches the Tinker training algorithm that connects to a separate store
    and rollout runners.
    """
    config = Config(
        learning_rate=1e-5,
        dataset_builder=AGLDatasetBuilder(
            train_dataset=[str(i) for i in range(1000)],
            val_dataset=[str(i) for i in range(1000, 1024)],
            batch_size=32,
            shuffle=True,
            group_size=4,
            seed=42,
        ),
        renderer_name="qwen3",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        log_path="logs/hello",
        max_tokens=32,
        store_address="http://localhost:4747",
    )
    asyncio.run(entrypoint(config))


def run_rollout(*, worker_id: int) -> None:
    """Rollout runner, single-process."""
    tracer = agl.AgentOpsTracer()

    runner = agl.LitAgentRunner[str](tracer=tracer)

    console.print(f"[bold green]Runners:[/bold green] Rollout runner {worker_id} started.")

    store = agl.LightningStoreClient("http://localhost:4747")
    with runner.run_context(agent=hello, store=store, worker_id=worker_id):
        asyncio.run(runner.iter())


def spawn_runners(*, n_runners: int) -> None:
    """Spawn a set of rollout runners in separate processes.

    Args:
        n_runners: The number of runners to spawn.
    """

    runners = [
        multiprocessing.Process(target=run_rollout, kwargs={"worker_id": worker_id}) for worker_id in range(n_runners)
    ]
    for runner in runners:
        runner.start()

    for runner in runners:
        runner.join()


def oneclick():
    """Run integrated training with algorithm and runners in one process.

    This is the simplest way to run the example, as it handles spawning
    the store, algorithm, and runners automatically.
    """
    config = Config(
        learning_rate=1e-5,
        dataset_builder=AGLDatasetBuilder(
            batch_size=16,
            group_size=4,
            seed=42,
            n_epochs=1,
        ),
        renderer_name="qwen3",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        log_path="logs/hello",
        max_tokens=32,
        llm_proxy_port=_find_available_port(),
    )
    trainer = agl.Trainer(
        algorithm=Tinker(config),
        llm_proxy=agl.LLMProxy(port=12306, num_retries=3),
        n_runners=8,
        port=_find_available_port(),
    )
    trainer.fit(hello, train_dataset=[str(i) for i in range(1000)], val_dataset=[str(i) for i in range(1000, 1024)])


def main():
    """Entry point for the hello example script."""
    parser = argparse.ArgumentParser(description="Train a hello echo agent with Agent-lightning + Tinker.")
    parser.add_argument("mode", type=str, choices=["algo", "runner", "oneclick"])

    args = parser.parse_args()

    agl.configure_logger()
    if args.mode == "algo":
        run_algo()
    elif args.mode == "runner":
        spawn_runners(n_runners=8)
    elif args.mode == "oneclick":
        oneclick()


if __name__ == "__main__":
    main()
