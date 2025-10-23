# Copyright (c) Microsoft. All rights reserved.

from typing import cast

import litellm
import openai
import tinker
from agl_tinker.llm import TinkerLLM
from litellm.utils import custom_llm_setup
from rich.console import Console
from tinker_cookbook.renderers import Qwen3Renderer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agentlightning import InMemoryLightningStore, LLMProxy, configure_logger

configure_logger(name="agentlightning")
configure_logger(name="agl_tinker", level=logging.INFO)


def main():
    console = Console()
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name))  # type: ignore
    renderer = Qwen3Renderer(tokenizer)  # type: ignore
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tinker_llm = TinkerLLM(
        renderer=renderer, tokenizer=tokenizer, sampling_client=sampling_client, default_max_tokens=20
    )

    litellm.custom_provider_map = [{"provider": "agl-tinker", "custom_handler": tinker_llm}]

    custom_llm_setup()

    store = InMemoryLightningStore()
    llm_proxy = LLMProxy(
        port=4000,
        store=store,
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {"model": f"agl-tinker/{model_name}"},
            }
        ],
        num_retries=0,
    )

    try:
        console.print("Starting LLM proxy...")
        llm_proxy.start()
        console.print("LLM proxy started")

        client = openai.OpenAI(base_url="http://localhost:4000/v1", api_key="dummy")
        response = client.chat.completions.create(
            model="Qwen3-30B-A3B-Instruct-2507",
            messages=[{"role": "user", "content": "Hello world!"}],
        )
        print(response)
    finally:
        console.print("Stopping LLM proxy...")
        llm_proxy.stop()
        console.print("LLM proxy stopped")


if __name__ == "__main__":
    main()
