# Copyright (c) Microsoft. All rights reserved.

import pandas as pd
from aoai_finetune import AzureOpenAIFinetune
from capital_agent import capital_agent
from rich.console import Console

from agentlightning import TraceToMessages, Trainer, configure_logger

finetune_algo = AzureOpenAIFinetune(
    base_deployment_name="gpt-4.1-mini",
    finetuned_deployment_name="gpt-4.1-mini-ft",
    base_model_name="gpt-4.1-mini-2025-04-14",
    finetune_every_n_rollouts=24,
    data_filter_ratio=0.6,
)
# finetune_algo._deploy_model(
#     model_name="gpt-4.1-mini-2025-04-14.ft-071a9d9c59ec4d088d1a3e56707d7361-aoai_ft_1",
#     version="2"
# )

configure_logger()
# finetune_algo._wait_for_deployment_ready("gpt-4.1-mini-ft", "1")
# finetune_algo.deploy_finetuned_model("gpt-4.1-mini-2025-04-14.ft-071a9d9c59ec4d088d1a3e56707d7361-aoai_ft_1", 2)

finetune_algo._delete_deployment("gpt-4.1-mini-ft_v01")
