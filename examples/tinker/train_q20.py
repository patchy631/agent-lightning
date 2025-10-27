import os
import traceback
from typing import Any, TypedDict, cast

import pandas as pd
from agl_tinker.llm import create_llm_proxy
from crewai import LLM as CrewLLM
from rich.console import Console
from twenty_questions import AnswererResponse, SearchTool, TwentyQuestionsFlow

import agentlightning as agl


class Q20Task(TypedDict):
    category: str
    answer: str
    search_enabled: bool


LLM_TIMEOUT = 120.0

console = Console()


@agl.rollout
async def q20_agent(task: Q20Task, llm: agl.LLM, rollout: agl.Rollout) -> None:
    player_llm = CrewLLM(model="openai/" + llm.model, base_url=llm.endpoint, api_key="dummy", timeout=LLM_TIMEOUT)
    answer_llm = CrewLLM(
        model="openai/gpt-5-mini",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort="low",
        response_format=AnswererResponse,
        timeout=LLM_TIMEOUT,
    )
    if task["search_enabled"]:
        search_tool = SearchTool(
            model=CrewLLM(
                model="openai/gpt-4.1",
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                reasoning_effort="none",
                timeout=LLM_TIMEOUT,
            )
        )
    else:
        search_tool = None

    flow = TwentyQuestionsFlow(player_llm=player_llm, answer_llm=answer_llm, search_tool=search_tool)
    try:
        await flow.kickoff_async(cast(Any, task))
        agl.emit_reward(1.0 if flow.state.correct else 0.0)
    except Exception as e:
        console.print(f"Error in q20_agent: {traceback.format_exc()}")
        agl.emit_exception(e)
        agl.emit_reward(0.0)


def dry_run():
    store = agl.InMemoryLightningStore()
    llm_proxy = create_llm_proxy("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_instruct", store=store)
    trainer = agl.Trainer(
        n_runners=2,
        initial_resources={"llm": llm_proxy.as_resource()},
        store=store,
    )
    try:
        llm_proxy.start()
        sampled_csv = pd.read_csv("twenty_questions_nouns.csv").sample(n=4, random_state=42)  # type: ignore
        sampled_csv["search_enabled"] = False
        dataset = sampled_csv.to_dict(orient="records")  # type: ignore
        trainer.dev(q20_agent, cast(agl.Dataset[Q20Task], dataset))
    finally:
        llm_proxy.stop()


if __name__ == "__main__":
    agl.configure_logger()
    dry_run()
