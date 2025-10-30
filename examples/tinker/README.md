# Tinker + Agent-lightning Integration

This example demonstrates how to leverage Tinker's training/sampling service as a fine-tunable LLM backend for agents, under the hood using Agent-lightning.

## How it's different from the original Tinker Cookbook's RL example?

Real-world agent applications orchestrate agents either with agent frameworks like LangChain, OpenAI Agent SDK, AutoGen, CrewAI, Microsoft Agent Framework...; or without those frameworks by directly calling `openai.chat.completion` APIs. They usually look like this:

```python
def guess_number_agent():
    client = openai.OpenAI()
    messages = [{"role": "system", "content": "Guess a number between 1 and 100."}]
    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
        )
        response_content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_content})
        guessed_number = extract_number(response_content)
        if guessed_number == gold_answer:
            return 1.0
        elif guessed_number < gold_answer:
            messages.append({"role": "user", "content": "Too low"})
        else:
            messages.append({"role": "user", "content": "Too high"})
    return 0.0
```

but to train the agent above with the original Cookbook code, you need to rewrite the agent into a callback-env style (the following example code is simplified. the real code is even more complex):

```python
class GuessNumberEnv:
    def __init__(self, gold_answer: int):
        self.system_prompt: Message = {"role": "system", "content": SYSTEM_PROMPT}
        self.turns: list[Message] = []
        self.gold_answer: int = gold_answer

    async def initial_observation(self) -> list[int]:
        """Return the initial observation in tokenized form."""
        return message_to_tokens(self.system_prompt)

    async def step(self, action_tokens: list[int]) -> tuple[list[int], float, bool]:
        """Accepts the action/generation tokens and returns the next prompt tokens."""
        action_message = tokens_to_message(action_tokens)
        guessed_number = extract_number(action_message["content"])

        if guessed_number == self.gold_answer:
            text, reward = "Correct", 1.0
        elif guessed_number < self.gold_answer:
            text, reward = "Too low", 0.0
        else:
            text, reward = "Too high", 0.0

        self.turns.append(action_message)
        self.turns.append({"role": "assistant", "content": text})
        episode_done = (reward == 1) or (len(self.turns) // 2 >= MAX_TURNS)

        return message_to_tokens(self.turns), reward, episode_done
```

The two code snippets above are essentially doing the same thing, but the first one is more convenient for agent development, while the second one is more convenient for training. The role of Agent-lightning is to provide a sophisticated middleware and store interface that allows you to write your agent in the first style, while also make it trainable.

## Included Files

| File/Directory | Description |
| -------------- | ----------- |
| `hello.py`     | ...         |

...

## Installation

Use the following command if you are installing agent-lightning from source:

```bash
uv sync --frozen --extra apo --group dev --group agents --group tinker
```

Otherwise, please install `tinker` and `tinker_cookbook` manually. See [Tinker's official repository](https://github.com/thinking-machines-lab/tinker-cookbook) for more details.

## Example 1: Hello 1024

How to run the hello 1024 example?

## Example 2: 20 Questions

How to run the 20 questions example?

## How it works

built upon `tinker_cookbook.rl`.
replace the env with a dummy shell with only task.
replace the `do_single_rollout` with enqueue tasks to Agent-lightning store and retrieve trajectory from spans.
wrap Tinker's sampling client with a [LiteLLM's CustomLLM](https://docs.litellm.ai/docs/providers/custom_llm_server) so that it exposes an OpenAI-compatible API via LiteLLM for agents to use.

...
