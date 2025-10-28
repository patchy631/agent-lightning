# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from typing import Any, Optional

import chz

from agentlightning.adapter import TracerTraceToTriplet
from agentlightning.algorithm import Algorithm
from agentlightning.llm_proxy import LLMProxy
from agentlightning.types import Dataset

from .train import Config, main_training_loop

logger = logging.getLogger(__name__)


class Tinker(Algorithm):

    def __init__(self, config: Config) -> None:
        self.config = config

    async def run(
        self, train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None
    ) -> None:
        if train_dataset is None or val_dataset is None:
            raise ValueError("train_dataset and val_dataset are required")

        config = chz.replace(  # type: ignore
            self.config,
            dataset_builder=chz.replace(  # type: ignore
                self.config.dataset_builder, train_dataset=train_dataset, val_dataset=val_dataset
            ),
        )

        store = self.get_store()
        adapter = self.get_adapter()
        if not isinstance(adapter, TracerTraceToTriplet):
            raise ValueError("Adapter must be a TracerTraceToTriplet")
        llm_proxy = self.get_llm_proxy()
        if llm_proxy is None:
            logger.warning("No LLM proxy found, creating one for you.")

            llm_proxy = LLMProxy(
                port=config.llm_proxy_port,
                model_list=[],
                store=store,
            )

        await main_training_loop(config, store, adapter, llm_proxy)
