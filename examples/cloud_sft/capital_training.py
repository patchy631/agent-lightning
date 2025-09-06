import os
from typing import cast

import openai
import pandas as pd
from capital_tool_use import run_task
from cloud_finetune_endpoint import AzureOpenAIFinetuneEndpoint

from agentlightning import LitAgent, LLM, Trainer, configure_logger

configure_logger()


class LitCapitalAgent(LitAgent):

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

    def training_rollout(self, task, rollout_id, resources):
        llm: LLM = cast(LLM, resources["main_llm"])
        openai_client = openai.OpenAI(api_key=self.api_key, base_url=llm.endpoint)

        return run_task(openai_client, llm.model, task)


if __name__ == "__main__":
    trainer = Trainer(n_workers=1)  # only 1 is supported currently
    tasks = pd.read_csv("capital_samples.csv").to_dict(orient="records")
    endpoint = AzureOpenAIFinetuneEndpoint(
        tasks=tasks,
        # base_deployment_name="gpt-4o-mini",
        # deployment_name="gpt-4o-mini",
        base_deployment_name="gpt-4o",
        deployment_name="gpt-4o-new",
        finetune_every_n_tasks=10,
    )
    trainer.fit(LitCapitalAgent(), endpoint)
    # endpoint._deploy_model("gpt-4o-2024-08-06.ft-9f4f6856285843c992825bb720835c1d", "2")
