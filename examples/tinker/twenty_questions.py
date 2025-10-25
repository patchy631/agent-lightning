from typing import Any, List, Literal, Optional, TypedDict, cast

from crewai import LLM as CrewLLM
from crewai import Agent as CrewAgent
from crewai import BaseLLM
from crewai import Task as CrewTask
from crewai.flow import Flow, listen, router, start
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console

llm = CrewLLM(model="openai/gpt-4o-mini")

console = Console()


class AnswererResponse(BaseModel):
    # Keep this short; do NOT ask for chain-of-thought
    brief_reason: Optional[str] = Field(description="1-2 sentences justification (optional, high level only).")
    yes_or_no: bool = Field(description="Whether the correct answer to the player's question is yes or no.")
    correct: bool = Field(
        description="Whether the player has correctly guessed the entity, and the game should end now."
    )


PLAYER_QUERY_TEMPLATE = """You are playing 20 Questions as the **Player**.
Ask one high-information **yes/no** question that most reduces the remaining possibility space.
If you think you have figured out the secret entity, ask a direct guess in the form: **"Is it <entity>?"**

THIS IS TURN #{turn_index} OF 20. You have {remaining_turns} turns left.

## What you have: Game history (Q/A pairs):

{history}

## Important assumptions

- All secret entities are **straightforward, familiar, and commonly known** (e.g., "Apple", "Paris", "Tiger").
- No complex, conditional, or composite answers. For example, the secret entity will **never** be something like “A door used in a haunted house” or “A Fender-produced guitar.”
- Each answer refers to a **single, clear concept** — not a variant, version, or situation-dependent form.

## How to decide your next question

- Use binary-splitting logic: prefer questions that partition the remaining candidates roughly in half.
- Start broad then focus (category -> traits -> unique identifiers).
- Take the remaining turns into account. If you have only one turn left, ask a direct guess.
- Do **not** ask about entities that directly name or define the answer (e.g., "Is it a type of pizza?" if "Pizza" is an option).
- Avoid questions that depend on subjective or situational conditions (e.g., "Would most people consider it artistic?").
- Avoid redundant, overlapping, or trivially true/false questions.
- If you use the search tool, do so only to verify factual implications behind your candidate question; do **not** paste search results. Think privately, then output just the question.

## Output format (critical)

- Output **only** one yes/no question on a single line.
- No preamble, no numbering, no quotes, no meta commentary.
- Keep it concise, under 50 words.
- If guessing: use the form **Is it <entity>?**

Now produce your single best next question."""


ANSWERER_QUERY_TEMPLATE = """You are the **Answerer** in 20 Questions. Your secret entity is: "{answer}".

## Game history

{history}

## The player's current question

{next_question}

## Rules

- Respond only with a structured yes/no evaluation about the entity.
- Be concise, objective, and consistent with previous answers.
- Never reveal the entity unless the player guessed correctly.

### Handling ambiguous entities

If the secret entity has multiple common meanings (e.g., "football" can mean both the **sport** and the **ball**):
- Answer **"yes"** if the question is true for **any major, well-recognized meaning** that does **not contradict** earlier answers.
- Answer **"no"** only if the question is false for **all reasonable interpretations** or if saying "yes" would **conflict** with prior responses.
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

    async def _run(self, search_query: str) -> str:
        """Perform a mocked search request using an LLM."""

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
    response: bool


class TwentyQuestionsGameState(BaseModel):
    answer: str = ""
    category: str = ""
    next_question: str = ""
    correct: bool = False
    turn_index: int = 1
    interactions: List[Turn] = Field(default_factory=list)

    def render_history(self) -> str:
        return "\n\n".join(
            [
                f"Question #{i}: {turn.question}\nResponse #{i}: {'yes' if turn.response else 'no'}"
                for i, turn in enumerate(self.interactions, start=1)
            ]
        )


class TwentyQuestionsFlow(Flow[TwentyQuestionsGameState]):

    def __init__(self, *args: Any, **kwargs: Any):
        self.player_llm = cast(CrewLLM, kwargs.pop("player_llm"))
        self.answer_llm = cast(CrewLLM, kwargs.pop("answer_llm"))
        self.search_tool = cast(Optional[SearchTool], kwargs.pop("search_tool", None))
        super().__init__(*args, **kwargs)

    @start("next_turn")
    async def ask_question(self):
        print("Starting flow")
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
        )

        result = await agent.kickoff_async(query)
        console.print(f"[bold red]Player (Turn {self.state.turn_index}):[/bold red] {result.raw}")
        self.state.next_question = result.raw

    @listen(ask_question)
    async def answer_question(self):
        query = ANSWERER_QUERY_TEMPLATE.format(
            answer=self.state.answer, next_question=self.state.next_question, history=self.state.render_history()
        )
        answerer_response = cast(AnswererResponse, self.answer_llm.call(query))  # type: ignore
        console.print(f"[bold red]Answerer (Turn {self.state.turn_index}):[/bold red] {answerer_response}")
        self.state.interactions.append(Turn(question=self.state.next_question, response=answerer_response.yes_or_no))
        self.state.next_question = ""  # Reset the next question

        if answerer_response.correct:
            self.state.correct = True
        else:
            self.state.correct = False

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


flow = TwentyQuestionsFlow(
    player_llm=CrewLLM(model="openai/gpt-4.1-mini"),
    answer_llm=CrewLLM(model="openai/gpt-5-mini", reasoning_effort="low", response_format=AnswererResponse),
    search_tool=SearchTool(model=CrewLLM(model="openai/gpt-4.1-mini")),
)
flow.plot()
try:
    result = flow.kickoff(
        {
            "answer": "football",
            # "answer": "Violin",
            "category": "person",
        }
    )
except Exception as e:
    raise

# search_tool = SearchTool(model=CrewLLM(model="openai/gpt-4.1-mini"))
# result = search_tool.run("What is the capital of France?")
# print(result)
