import json
import traceback
from typing import Any, List, Literal, Optional, TypedDict, cast

import pandas as pd
from crewai import LLM as CrewLLM
from crewai import Agent as CrewAgent
from crewai import BaseLLM
from crewai import Task as CrewTask
from crewai.flow import Flow, listen, router, start
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rich.console import Console

llm = CrewLLM(model="openai/gpt-4o-mini")

console = Console()


class AnswererResponse(BaseModel):
    # Keep this short; do NOT ask for chain-of-thought
    brief_reason: Optional[str] = Field(description="1-2 sentences justification (optional, high level only).")
    yes_or_no: Literal["yes", "no", "n/a"] = Field(
        description="Whether the correct answer to the player's question is yes, no, or not applicable."
    )
    correct: bool = Field(
        description="Whether the player has correctly guessed the entity, and the game should end now."
    )


PLAYER_QUERY_TEMPLATE = """You are playing 20 Questions as the **Player**.
Ask one high-information **yes/no** question that most reduces the remaining possibility space.
If you think you have figured out the secret entity, ask a direct guess in the form: **"Is it <entity>?"**

THIS IS TURN #{turn_index} OF 20. You have {remaining_turns} turns left. The quicker you make a correct guess, the higher your score.

## Important assumptions

- All answers must belong to one of the following categories: {categories}.
- All answers are **straightforward, familiar, and commonly known**. They can be at most 3 words long (and only one word long in a majority of cases).
- Each answer refers to a **single, clear concept** — not a variant, version, or situation-dependent form.  They will **never** be something like "A door used in a haunted house" or "A Fender-produced guitar".

## What you have: Game history (Q/A pairs):

{history}

## How to decide your next question

- Use binary-splitting logic: prefer questions that partition the remaining candidates roughly in half.
- Start broad then focus (category -> traits -> unique identifiers).
- Take the remaining turns into account. If you have only one turn left, ask a direct guess.
- Do **not** ask about entities that directly name or define the answer (e.g., "Is it a type of pizza?" if "Pizza" is an option).
- The answerer may reply **"n/a"** when a yes/no would be meaningless or not publicly knowable. Use this to your advantage.
- Avoid questions that depend on subjective or situational conditions (e.g., "Would most people consider it artistic?").
- You are encourage to use the search tool to verify factual implications behind your candidate question. This will also help you thinking more deeply and avoiding asking irrelevant or trivial questions.

## Output format (critical)

- Output **only** one yes/no question on a single line.
- No preamble, no numbering, no quotes, no meta commentary.
- Keep it concise, under 50 words.
- If guessing: use the form **Is it <entity>?**

Now produce your single best next question."""


ANSWERER_QUERY_TEMPLATE = """You are the **Answerer** in 20 Questions. Answer yes/no questions truthfully about the secret entity; mark correct if guessed exactly.

Your secret entity is: "{answer}".

## The player's current question

{next_question}

## Rules

- Respond only with a structured yes/no evaluation about the entity.
- Be concise, objective, and consistent with previous answers.
- Never reveal the entity unless the player guessed correctly.
- If you don't know the answer, for example, the information is never publicly known, or the question is irrelevant to the secret entity, answer **"n/a"**.

### Handling unknown or irrelevant questions

- If the question asks about something that is **not publicly known**, **not factual**, **ambiguous**, or **irrelevant** to the entity's nature (e.g., "Does it enjoy music?" for *Mount Everest*), respond with **"n/a"**.
- Use **"n/a"** only when a yes/no answer would be **misleading or nonsensical**.
- Examples:
  - "Does it have parents?" -> *n/a* (not meaningful for a place or object)
  - "Is it alive?" -> valid for all entities (answer yes/no if possible)
  - "Is it an animal?" -> *n/a* if the entity is a person, as this can be ambiguous.
  - "Does it post on social media?" -> *n/a* unless the entity is a living or fictional character known for doing so.
  - "Is the chair branded by a famous manufacturer?" -> *n/a* for a general object like "chair".

### Handling ambiguous entities

If the secret entity has multiple common meanings (e.g., "football" can mean both the **sport** and the **ball**):
- Answer **"yes"** if the question is true for **any** of the major, well-recognized meanings.
- Answer **"no"** only if the question is false for **all reasonable interpretations**.
- Avoid overinterpreting rare or niche meanings — stick to mainstream, widely understood ones.

### Handling direct guesses

If the player's question is a direct guess ("Is it ...?"):
- Set **correct = true** if the guess is a close match in meaning to the secret entity (e.g., “Is it cell phone?” ≈ “Smartphone”).
- Otherwise, set **correct = false**.
"""


SEARCH_PROMPT_TEMPLATE = """You are simulating a web search.

Query: "{search_query}"

Return a concise, factual summary (2-4 sentences) of the most relevant information you would find online.
Avoid speculation, filler, or references to being an AI. Just give the facts."""


class SearchToolInput(BaseModel):
    """Schema for search tool input."""

    search_query: str = Field(
        ...,
        description="A short, factual query describing what to search for (e.g., 'capital of France', 'biography of Ada Lovelace').",
    )


class SearchTool(BaseTool):
    """A mock web search tool powered by an LLM.

    This class mimics a real search engine call by using a lightweight LLM model.
    It can later be replaced by a real API (like Serper or Bing) without changing its interface.
    """

    model: BaseLLM
    name: str = "search"
    description: str = (
        "Search the web (mocked). Provide a concise, factual summary of what is known about the given topic."
    )
    num_called: int = 0

    async def _run(self, search_query: str) -> str:
        """Perform a mocked search request using an LLM."""
        self.num_called += 1
        # Safety: ensure input is not too long or empty
        search_query = search_query.strip()
        if not search_query:
            return "No query provided."
        if len(search_query) > 500:
            search_query = search_query[:500] + "..."

        # Use a lightweight CrewAgent to simulate a factual web summary
        agent = CrewAgent(
            role="Search engine summarizer",
            goal=(
                "Given a user's search query, return a concise, factual summary "
                "as if retrieved from reliable sources. "
                "Act like a real search engine summarizer. "
                "Never disclose that you are a simulator of a search engine."
            ),
            backstory=(
                "You simulate a web search engine, producing factual, neutral summaries. "
                "Do not fabricate sources or URLs. Focus on core, verifiable facts."
            ),
            llm=self.model,
        )

        prompt = SEARCH_PROMPT_TEMPLATE.format(search_query=search_query)
        result = await agent.kickoff_async(prompt)
        return result.raw.strip()


class Turn(BaseModel):
    question: str
    response: Literal["yes", "no", "n/a"]


class TwentyQuestionsGameState(BaseModel):
    answer: str = ""
    category: str = ""
    correct: bool = False
    num_tool_calls: int = 0
    next_question: str = ""
    turn_index: int = 1
    interactions: List[Turn] = Field(default_factory=list)

    def render_history(self) -> str:
        return "\n\n".join(
            [
                f"Question #{i}: {turn.question}\nResponse #{i}: {turn.response}"
                for i, turn in enumerate(self.interactions, start=1)
            ]
        )


class TwentyQuestionsFlow(Flow[TwentyQuestionsGameState]):

    def __init__(self, *args: Any, **kwargs: Any):
        self.player_llm = cast(CrewLLM, kwargs.pop("player_llm"))
        self.answer_llm = cast(CrewLLM, kwargs.pop("answer_llm"))
        self.search_tool = cast(Optional[SearchTool], kwargs.pop("search_tool", None))
        self.categories = cast(List[str], kwargs.pop("categories"))
        super().__init__(*args, **kwargs)

    @start("next_turn")
    async def ask_question(self):
        agent = CrewAgent(
            role="Player in a game of 20 questions",
            goal="Minimize uncertainty and identify the hidden entity within 20 yes/no questions.",
            backstory="A focused reasoner who uses binary-partition questions and only outputs one concise yes/no question per turn.",
            tools=[self.search_tool] if self.search_tool else [],
            llm=self.player_llm,
        )
        query = PLAYER_QUERY_TEMPLATE.format(
            history=self.state.render_history(),
            turn_index=self.state.turn_index,
            remaining_turns=20 - self.state.turn_index + 1,
            categories=", ".join(self.categories),
        )

        result = await agent.kickoff_async(query)
        console.print(f"[bold red]Player (Turn {self.state.turn_index}):[/bold red] {result.raw}")
        if self.search_tool is not None:
            self.state.num_tool_calls = self.search_tool.num_called
        self.state.next_question = result.raw

    @listen(ask_question)
    async def answer_question(self):
        query = ANSWERER_QUERY_TEMPLATE.format(answer=self.state.answer, next_question=self.state.next_question)
        answerer_response = cast(AnswererResponse, self.answer_llm.call(query))  # type: ignore
        console.print(f"[bold red]Answerer (Turn {self.state.turn_index}):[/bold red] {answerer_response}")
        try:
            turn = Turn(question=self.state.next_question, response=answerer_response.yes_or_no)
            correct = answerer_response.correct
        except Exception as e:
            console.print(f"[bold red]Answerer Response Format Error: {e}[/bold red]")
            # Assuming n/a
            turn = Turn(question=self.state.next_question, response="n/a")
            correct = False
        self.state.interactions.append(turn)
        self.state.next_question = ""  # Reset the next question
        self.state.correct = correct

    @router(answer_question)
    def game_should_continue(self):
        if self.state.correct:
            console.print(f"[bold red]Correct! You win![/bold red]")
            return "game_over"
        elif self.state.turn_index >= 20:
            console.print(
                f"[bold red]You've asked 20 questions and still haven't guessed the entity. You lose![/bold red]"
            )
            return "game_over"
        else:
            self.state.turn_index += 1
            console.print(f"[bold purple]Continue with turn #{self.state.turn_index}...[/bold purple]")
            return "next_turn"

    @listen("game_over")
    def finish(self):
        console.print("The flow has reached the finished state.")


def main():
    df = pd.read_csv("twenty_questions_nouns.csv")  # type: ignore
    categories = cast(List[str], df["category"].unique().tolist())  # type: ignore

    for index, row in df.sample(n=len(df)).iterrows():  # type: ignore
        flow = TwentyQuestionsFlow(
            player_llm=CrewLLM(model="openai/gpt-4.1", timeout=60.0),
            answer_llm=CrewLLM(
                model="openai/gpt-5-mini", reasoning_effort="low", response_format=AnswererResponse, timeout=60.0
            ),
            search_tool=SearchTool(model=CrewLLM(model="openai/gpt-4.1-mini", timeout=60.0)),
            categories=categories,
        )
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
        with open("logs/twenty_questions.jsonl", "a") as f:
            f.write(json.dumps(result_json) + "\n")


if __name__ == "__main__":
    main()
