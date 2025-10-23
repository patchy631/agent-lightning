# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from typing import cast

import litellm
import openai
import tinker
from agl_tinker.llm import TinkerLLM
from rich.console import Console
from tinker_cookbook.renderers import Qwen3Renderer
from transformers import AutoTokenizer, PreTrainedTokenizer

from agentlightning import AgentOpsTracer, InMemoryLightningStore, LLMProxy, configure_logger

configure_logger(name="agentlightning")
configure_logger(name="agl_tinker", level=logging.INFO)


async def main():
    console = Console()
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_name))  # type: ignore
    renderer = Qwen3Renderer(tokenizer)  # type: ignore
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tinker_llm = TinkerLLM(
        model_name=model_name, renderer=renderer, tokenizer=tokenizer, sampling_client=sampling_client, max_tokens=20
    )
    tinker_llm.rewrite_litellm_custom_providers()

    store = InMemoryLightningStore()
    rollout = await store.start_rollout("dummy", "train")
    llm_proxy = LLMProxy(
        port=4000,
        store=store,
        model_list=tinker_llm.as_model_list(),
        num_retries=0,
    )

    try:
        console.print("Starting LLM proxy...")
        llm_proxy.start()
        console.print("LLM proxy started")

        tracer = AgentOpsTracer()
        client = openai.OpenAI(
            base_url=f"http://localhost:4000/rollout/{rollout.rollout_id}/attempt/{rollout.attempt.attempt_id}",
            api_key="dummy",
        )

        tracer.init()
        tracer.init_worker(0)
        async with tracer.trace_context(
            name="test_llm", store=store, rollout_id=rollout.rollout_id, attempt_id=rollout.attempt.attempt_id
        ):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello world!"}],
                max_tokens=10,
                temperature=0.5,
                top_p=0.9,
                seed=43,
            )
            print(response)
        tracer.teardown_worker(0)
        tracer.teardown()

        for store_span in await store.query_spans(rollout.rollout_id):
            print(store_span)
    finally:
        console.print("Stopping LLM proxy...")
        llm_proxy.stop()
        console.print("LLM proxy stopped")


if __name__ == "__main__":
    asyncio.run(main())
