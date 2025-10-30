# Copyright (c) Microsoft. All rights reserved.

"""Standalone 20 Questions game evaluation script.

This script evaluates a language model on a 20 Questions game where the model
acts as the player trying to guess a secret entity by asking yes/no questions.

To run an evaluation:

```bash
# Set environment variables
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_base_url

# Run evaluation on a Qwen model
python q20_agent.py --model Qwen/Qwen3-30B-A3B-Instruct-2507

# Run with search tool enabled
python q20_agent.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --search

# Run on OpenAI model
python q20_agent.py --model gpt-4o-mini
```

The script outputs results to a JSONL file for analysis.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from agl_tinker.llm import create_llm_proxy
from crewai import LLM as CrewLLM
from q20_agent import AnswererResponse, SearchTool, TwentyQuestionsFlow
from rich.console import Console

from agentlightning import InMemoryLightningStore, LLMProxy

console = Console()


def evaluate_q20(
    model_name: str,
    search: bool,
    port: int,
    output_file: str,
    dataset_path: str,
    seed: Optional[int] = 42,
):
    """Evaluate a model on the 20 Questions game.

    Args:
        model_name: The name of the model to evaluate.
        search: Whether the player can use the search tool.
        port: The port to use for the LiteLLM proxy.
        output_file: Where to append JSONL results.
        dataset_path: CSV file containing category and answer columns.
        seed: Optional random seed for shuffling the dataset; ``None`` disables deterministic shuffling.
    """

    store = InMemoryLightningStore()
    df = pd.read_csv(dataset_path)  # type: ignore
    if df.empty:
        console.print(f"[bold yellow]Dataset '{dataset_path}' is empty. Nothing to evaluate.[/bold yellow]")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_name.startswith("Qwen/"):
        llm_proxy = create_llm_proxy(model_name, "qwen3", port, store, _add_return_token_ids=False)
    else:
        console.print(f"Assuming {model_name} is an OpenAI model.")
        llm_proxy = LLMProxy(
            port=port,
            store=store,
            model_list=[
                {"model_name": model_name, "litellm_params": {"model": "openai/" + model_name}},
            ],
            num_retries=2,
            _add_return_token_ids=False,
        )

    answerer_model_name = "gpt-5-mini"
    search_model_name = "gpt-4.1"

    # Add the answerer and search models to the model list if they are not already present.
    current_model_list = llm_proxy.model_list.copy()
    if not any(model["model_name"] == answerer_model_name for model in current_model_list):
        current_model_list.append(
            {"model_name": answerer_model_name, "litellm_params": {"model": "openai/" + answerer_model_name}}
        )
    if not any(model["model_name"] == search_model_name for model in current_model_list):
        current_model_list.append(
            {"model_name": search_model_name, "litellm_params": {"model": "openai/" + search_model_name}}
        )
    llm_proxy.update_model_list(current_model_list)
    console.print("Model list:", llm_proxy.model_list)

    try:
        llm_proxy.start()
        player_llm = CrewLLM(
            model="openai/" + model_name, base_url=f"http://localhost:{port}/v1", api_key="dummy", timeout=60.0
        )
        answer_llm = CrewLLM(
            model="openai/" + answerer_model_name,
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            reasoning_effort="low",
            response_format=AnswererResponse,
        )
        search_tool = (
            SearchTool(
                model=CrewLLM(
                    model="openai/" + search_model_name,
                    base_url=f"http://localhost:{port}/v1",
                    api_key="dummy",
                    reasoning_effort="none",
                    timeout=60.0,
                )
            )
            if search
            else None
        )
        sampled_df = (
            df.sample(n=len(df), random_state=seed)  # type: ignore
            if seed is not None
            else df.sample(n=len(df))  # type: ignore
        )
        for index, row in sampled_df.iterrows():  # type: ignore
            if search_tool:
                search_tool.num_called = 0

            flow = TwentyQuestionsFlow(player_llm=player_llm, answer_llm=answer_llm, search_tool=search_tool)
            try:
                flow.kickoff(
                    {
                        "answer": row["answer"],
                        "category": row["category"],
                    }
                )
                result_json: dict[str, Any] = {"index": index, **flow.state.model_dump()}
            except Exception as e:
                result_json = {
                    "index": index,
                    "answer": row["answer"],
                    "category": row["category"],
                    "error": str(e),
                    "exception": traceback.print_exc(),
                }
            with output_path.open("a") as f:
                f.write(json.dumps(result_json) + "\n")
    finally:
        llm_proxy.stop()


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the 20 Questions evaluation script.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(description="Evaluate a model on the 20 Questions benchmark.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model identifier to evaluate.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable the search tool for the player agent.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12358,
        help="Port to expose the LiteLLM proxy.",
    )
    parser.add_argument(
        "--output-file",
        default="logs/twenty_questions_results.jsonl",
        help="Path to write JSONL evaluation results.",
    )
    parser.add_argument(
        "--dataset",
        default="q20_nouns.csv",
        help="CSV file containing the evaluation dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling the dataset. Use -1 to disable deterministic shuffling.",
    )

    args = parser.parse_args(argv)
    evaluate_q20(
        model_name=args.model,
        search=args.search,
        port=args.port,
        output_file=args.output_file,
        dataset_path=args.dataset,
        seed=None if args.seed == -1 else args.seed,
    )


if __name__ == "__main__":
    main()
