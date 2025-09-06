from agentlightning.types import LLM, Rollout
from agentlightning.client import DevTaskLoader


class AzureOpenAIFinetuneEndpoint(DevTaskLoader):

    def __init__(
        self,
        tasks: Union[List[TaskInput], List[Task]],
        model_name: str,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
        finetune_every_n_tasks: int = 10,
        **kwargs,
    ): ...

    def initial_llm(self) -> LLM:
        return LLM(endpoint="")

    def finetune(self, data: list[Rollout]) -> LLM: ...
