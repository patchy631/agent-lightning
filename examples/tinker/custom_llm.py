import litellm
from litellm import CustomLLM, completion, get_llm_provider
from litellm.utils import custom_llm_setup


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore


my_custom_llm = MyCustomLLM()

litellm.custom_provider_map = [{"provider": "my-custom-llm", "custom_handler": my_custom_llm}]

custom_llm_setup()

from agentlightning import InMemoryLightningStore
from agentlightning.llm_proxy import LLMProxy

store = InMemoryLightningStore()
llm_proxy = LLMProxy(
    port=4000,
    store=store,
    model_list=[{"model_name": "my-custom-model", "litellm_params": {"model": "my-custom-llm/my-model"}}],
)

llm_proxy.start()
import time

time.sleep(1000)
