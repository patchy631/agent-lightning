import json
import os
from typing import Any, Dict, List, TypedDict

import openai
import pandas as pd
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from rich.console import Console

from agentlightning import LLM, rollout

CAPITALS = {
    "japan": "Tokyo",
    "france": "Paris",
    "canada": "Ottawa",
    "australia": "Canberra",
    "brazil": "Brasília",
    "egypt": "Cairo",
    "kenya": "Nairobi",
    "spain": "Madrid",
    "italy": "Rome",
    "germany": "Berlin",
    "south korea": "Seoul",
    "india": "New Delhi",
}

console = Console()


def country_capital_lookup(country: str) -> str:
    return CAPITALS.get(country.strip().lower(), "Unknown")


class CapitalTask(TypedDict):
    input: str
    output: str


TOOLS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "country_capital_lookup",
            "description": "Get the capital city of a given country.",
            "parameters": {"type": "object", "properties": {"country": {"type": "string"}}, "required": ["country"]},
        },
    }
]

SYSTEM = (
    "You are a concise assistant. "
    "If the user asks for a country's capital, ALWAYS call the tool 'country_capital_lookup'. "
    "Otherwise, answer briefly."
)


@rollout
def capital_agent(task: CapitalTask, llm: LLM) -> float:
    """Run one evaluation task with capital agent.

    Returns 1.0 if output contains expected substring, else 0.0.
    """
    print("[bold blue][run_task][/bold blue] Running task with input:", task)
    prompt = task["input"]
    expected = task["output"]

    openai_client = openai.OpenAI(base_url=llm.endpoint)

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    # --- Call #1 ---
    first = openai_client.chat.completions.create(
        model=llm.model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    print("[bold blue][run_task][/bold blue] First call response:", first)

    msg = first.choices[0].message

    if msg.tool_calls:
        assistant_tool_calls: List[ChatCompletionMessageFunctionToolCallParam] = []
        tool_results: List[ChatCompletionToolMessageParam] = []
        for tc in msg.tool_calls:
            if tc.type == "function" and tc.function.name == "country_capital_lookup":
                args = json.loads(tc.function.arguments or "{}")
                result = country_capital_lookup(args.get("country", ""))
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": assistant_tool_calls,
            }
        )
        messages.extend(tool_results)
        print("[bold blue][run_task][/bold blue] Messages after tool call:", messages)

        # --- Call #2 ---
        second = openai_client.chat.completions.create(
            model=llm.model,
            messages=messages,
            temperature=0,
        )
        print("[bold blue][run_task][/bold blue] Second call response:", second)
        final_text = second.choices[0].message.content or ""
    else:
        print("[bold blue][run_task][/bold blue] No tool calls made.")
        final_text = msg.content or ""

    final_text = final_text.strip()
    reward = 1.0 if expected.lower() in final_text.lower() else 0.0
    print(f"[bold blue][run_task][/bold blue] Final output: {final_text} | Reward: {reward}")
    return reward


if __name__ == "__main__":
    client = openai.OpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), base_url=os.getenv("AZURE_OPENAI_ENDPOINT"))

    data = pd.read_csv("capital_samples.csv")
    sample = data.iloc[0].to_dict()
    run_task(client, "gpt-4o-new", sample)
