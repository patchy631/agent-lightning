# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from agl_tinker.env import AGLDatasetBuilder
from agl_tinker.train import Config
from agl_tinker.train import main as training_loop

from agentlightning import configure_logger

configure_logger(name="agentlightning")
configure_logger(name="agl_tinker", level=logging.INFO)


def main():
    dataset_builder = AGLDatasetBuilder(
        train_dataset=[f"Hello {i}" for i in range(30)],
        batch_size=10,
        shuffle=True,
        group_size=4,
        seed=42,
    )

    config = Config(
        learning_rate=1e-4,
        dataset_builder=dataset_builder,
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        log_path="logs",
        max_tokens=20,
    )

    asyncio.run(training_loop(config))


if __name__ == "__main__":
    main()
