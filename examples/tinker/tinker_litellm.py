from typing import Any, List, TypeGuard

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import Choices
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import ModelResponse
from tinker_cookbook.renderers import Message, Qwen3Renderer, Renderer
from transformers.models.auto.tokenization_auto import AutoTokenizer


def mock_qwen3_token_in_token_out(tokens: list[int]) -> list[int]:
    return [6, 7, 8, 9, 10]


class TinkerLLM(CustomLLM):
    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer

    def _validate_message(self, messages: Any) -> TypeGuard[List[Message]]:
        return True
        # TODO: implement this
        raise NotImplementedError()

    def completion(self, **kwargs: Any) -> litellm.ModelResponse:
        messages = kwargs.pop("messages", None)
        if not self._validate_message(messages):
            raise ValueError("...")
        inputs = self.renderer.build_generation_prompt(messages)

        token_out = mock_qwen3_token_in_token_out(inputs)
        response, parse_success = self.renderer.parse_response(token_out)

        return ModelResponse(
            id="123",
            choices=[
                Choices(
                    message=LitellmMessage(
                        role=response["role"],
                        content=response["content"],
                    )
                )
            ],
        )


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
my_custom_llm = TinkerLLM(Qwen3Renderer(tokenizer))

litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
    {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
]

resp = litellm.completion(
    model="my-custom-llm/my-fake-model",
    messages=[{"role": "user", "content": "Hello world!"}],
)
print(resp)
