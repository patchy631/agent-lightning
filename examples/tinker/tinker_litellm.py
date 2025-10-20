from typing import Any, List, TypeGuard

import litellm
from litellm.llms.custom_llm import CustomLLM
from tinker_cookbook.renderers import Message, Qwen3Renderer, Renderer


def mock_qwen3_token_in_token_out(tokens: list[int]) -> list[int]:
    return [6, 7, 8, 9, 10]


class TinkerLLM(CustomLLM):
    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer

    def _validate_message(self, messages: Any) -> TypeGuard[List[Message]]:
        # TODO: implement this
        raise NotImplementedError()

    def completion(self, **kwargs: Any) -> litellm.ModelResponse:
        messages = kwargs.pop("messages", None)
        if not self._validate_message(messages):
            raise ValueError("...")
        inputs = self.renderer.build_generation_prompt(messages)


my_custom_llm = MyCustomLLM()

litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
    {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
]

resp = litellm.completion(
    model="my-custom-llm/my-fake-model",
    messages=[{"role": "user", "content": "Hello world!"}],
)

assert resp.choices[0].message.content == "Hi!"
print(resp)
