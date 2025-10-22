import uuid
from typing import Any, List, Literal, TypeGuard

import litellm
import tinker
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ChatCompletionMessageToolCall, ChatCompletionTokenLogprob
from litellm.types.utils import ChoiceLogprobs as LitellmChoiceLogprobs
from litellm.types.utils import Choices
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import ModelResponse
from litellm.utils import custom_llm_setup
from tinker.types import ModelInput, SampleResponse, SamplingParams
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import Qwen3Renderer, Renderer
from tinker_cookbook.renderers import ToolCall as TinkerToolCall
from transformers.models.auto.tokenization_auto import AutoTokenizer

from agentlightning.logging import configure_logger

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")

logger = configure_logger(name="agentlightning.tinker")


def generate_id(prefix: str) -> str:
    return prefix + str(uuid.uuid4())


class TinkerLLM(CustomLLM):
    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer

    def _validate_message(self, messages: Any) -> TypeGuard[List[TinkerMessage]]:
        return True
        # TODO: implement this
        raise NotImplementedError()

    def _validate_role(self, role: str) -> TypeGuard[Literal["assistant", "user", "system", "tool", "function"]]:
        return role in ["assistant", "user", "system", "tool", "function"]

    def _parse_tool_call(self, tool_call: TinkerToolCall) -> ChatCompletionMessageToolCall:
        if set(tool_call.keys()) != {"name", "args"}:
            logger.warning(f"Found unexpected tool call keys: {tool_call.keys()}")
        return ChatCompletionMessageToolCall(
            id=generate_id("tinker-tool-call-"),
            function={
                "name": tool_call["name"],
                "arguments": tool_call["args"],
            },
            type="function",
        )

    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        messages = kwargs.pop("messages", None)
        if not self._validate_message(messages):
            raise ValueError("...")
        return self.renderer.build_generation_prompt(messages)

    def _parse_response(self, response: SampleResponse) -> ModelResponse:
        choices: List[Choices] = []
        for seq in response.sequences:
            parsed_response, parse_success = self.renderer.parse_response(seq.tokens)
            if parse_success:
                role = parsed_response["role"]
                if not self._validate_role(role):
                    raise ValueError(f"Invalid role: {role}")
                content = parsed_response["content"]
                tool_calls = parsed_response.get("tool_calls", None)
                if tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in tool_calls]

                if seq.logprobs is not None:
                    logprobs = LitellmChoiceLogprobs(
                        content=[
                            ChatCompletionTokenLogprob(
                                token=str(token),
                                logprob=logprob,
                                top_logprobs=[],
                            )
                            for token, logprob in zip(seq.tokens, seq.logprobs)
                        ]
                    )
                else:
                    logprobs = None

                choices.append(
                    Choices(
                        message=LitellmMessage(role=role, content=content, tool_calls=tool_calls),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                    )
                )
            else:
                raise ValueError(f"Failed to parse response: {parsed_response}")
        return ModelResponse(id=generate_id("tinker-sampling-"), choices=choices)

    async def acompletion(self, **kwargs: Any) -> ModelResponse:
        model_input = self._prepare_model_input(**kwargs)
        params = SamplingParams(max_tokens=20, temperature=0.0, stop=self.renderer.get_stop_sequences())
        result = await sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)
        return self._parse_response(result)


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")

tinker_llm = TinkerLLM(Qwen3Renderer(tokenizer))

litellm.custom_provider_map = [{"provider": "agl-tinker", "custom_handler": tinker_llm}]

custom_llm_setup()


from agentlightning import InMemoryLightningStore
from agentlightning.llm_proxy import LLMProxy

store = InMemoryLightningStore()
llm_proxy = LLMProxy(
    port=4000,
    store=store,
    model_list=[
        {
            "model_name": "Qwen3-30B-A3B-Instruct-2507",
            "litellm_params": {"model": "agl-tinker/Qwen3-30B-A3B-Instruct-2507"},
        }
    ],
)

llm_proxy.start()

import openai

client = openai.OpenAI(base_url="http://localhost:4000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen3-30B-A3B-Instruct-2507",
    messages=[{"role": "user", "content": "Hello world!"}],
)
print(response)

llm_proxy.stop()
