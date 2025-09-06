from typing import Dict, Any
import json
import os

import pandas as pd
import openai

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


def country_capital_lookup(country: str) -> str:
    return CAPITALS.get(country.strip().lower(), "Unknown")


TOOLS = [
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


def run_task(openai_client: openai.OpenAI, task_input: Dict[str, str]) -> float:
    """
    Run one evaluation task.
    Returns 1.0 if output contains expected substring, else 0.0.
    """
    print("[run_task] Running task with input:", task_input)
    prompt = task_input["input"]
    expected = task_input["expected_substring"]

    # --- Call #1 ---
    first = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    print("[run_task] First call response:", first)

    msg = first.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    if tool_calls:
        messages.append(
            {"role": "assistant", "tool_calls": [tc.to_dict() for tc in tool_calls], "content": msg.content or ""}
        )
        for tc in tool_calls:
            if tc.function.name == "country_capital_lookup":
                args = json.loads(tc.function.arguments or "{}")
                result = country_capital_lookup(args.get("country", ""))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "country_capital_lookup",
                        "content": json.dumps({"capital": result}),
                    }
                )
        print("[run_task] Messages after tool call:", messages)
    else:
        messages.append({"role": "assistant", "content": msg.content or ""})

    # --- Call #2 ---
    second = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    print("[run_task] Second call response:", second)

    final_text = second.choices[0].message.content.strip()
    reward = 1.0 if expected.lower() in final_text.lower() else 0.0
    print(f"[run_task] Final output: {final_text} | Reward: {reward}")
    return reward


if __name__ == "__main__":
    client = openai.OpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), base_url=os.getenv("AZURE_OPENAI_ENDPOINT"))

    data = pd.read_csv("capital_samples.csv")
    sample = data.iloc[0].to_dict()
    run_task(client, sample)
