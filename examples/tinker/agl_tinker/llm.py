# Copyright (c) Microsoft. All rights reserved.

import logging
import uuid
from typing import Any, List, Literal, TypeGuard

import tinker
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ChatCompletionMessageToolCall, ChatCompletionTokenLogprob
from litellm.types.utils import ChoiceLogprobs as LitellmChoiceLogprobs
from litellm.types.utils import Choices
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import ModelResponse
from pydantic import TypeAdapter
from tinker.types import ModelInput, SampleResponse, SamplingParams
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers import ToolCall as TinkerToolCall
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def generate_id(prefix: str) -> str:
    return prefix + str(uuid.uuid4())


class TinkerLLM(CustomLLM):
    def __init__(
        self,
        *,
        renderer: Renderer,
        tokenizer: PreTrainedTokenizerBase,
        sampling_client: tinker.SamplingClient,
        default_max_tokens: int = 2048,
    ) -> None:
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.sampling_client = sampling_client
        self.default_max_tokens = default_max_tokens

    def _validate_messages(self, messages: Any) -> TypeGuard[List[TinkerMessage]]:
        TypeAdapter(List[TinkerMessage]).validate_python(messages)
        # Exception will be raised if validation fails
        return True

    def _validate_role(self, role: str) -> TypeGuard[Literal["assistant", "user", "system", "tool", "function"]]:
        if role not in ["assistant", "user", "system", "tool", "function"]:
            raise ValueError(f"Invalid role: {role}")
        return True

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
        if self._validate_messages(messages):
            return self.renderer.build_generation_prompt(messages)
        else:
            assert False, "This should never happen"

    def _parse_response(self, model_input: ModelInput, response: SampleResponse) -> ModelResponse:
        choices: List[Choices] = []
        for seq in response.sequences:
            if seq.logprobs is not None:
                token_strings: List[str] = self.tokenizer.batch_decode([token] for token in seq.tokens)  # type: ignore
                logprobs = LitellmChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=token,
                            logprob=logprob,
                            top_logprobs=[],
                        )
                        for token, logprob in zip(token_strings, seq.logprobs)
                    ]
                )
            else:
                logprobs = None

            parsed_response, parse_success = self.renderer.parse_response(seq.tokens)
            if parse_success:
                role = parsed_response["role"]
                if not self._validate_role(role):
                    assert False, "This should never happen"
                content = parsed_response["content"]
                tool_calls = parsed_response.get("tool_calls", None)
                if tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in tool_calls]
                choices.append(
                    Choices(
                        message=LitellmMessage(role=role, content=content, tool_calls=tool_calls),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
            else:
                logger.warning(f"Failed to parse response: {parsed_response}")
                # Go with the default path
                choices.append(
                    Choices(
                        message=LitellmMessage(role="assistant", content=parsed_response["content"]),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
        return ModelResponse(
            id=generate_id("tinker-sampling-"), choices=choices, prompt_token_ids=model_input.to_ints()
        )

    async def acompletion(self, **kwargs: Any) -> ModelResponse:
        print(kwargs)
        model_input = self._prepare_model_input(**kwargs)
        params = SamplingParams(max_tokens=20, temperature=0.0, stop=self.renderer.get_stop_sequences())
        result = await self.sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)
        return self._parse_response(model_input, result)
