from typing import List, Literal, Optional, TypedDict, cast

from crewai import LLM as CrewAILLM
from crewai import Agent as CrewAgent
from crewai import Task as CrewTask
from crewai.flow import Flow, listen, router, start
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console

llm = CrewAILLM(model="openai/gpt-4o-mini")

console = Console()


class AnswererResponse(BaseModel):
    thinking: str = Field(description="Your step-by-step reasoning process to answer the question.")
    yes_or_no: bool = Field(description="Whether the answer to that question is yes or no.")
    correct: bool = Field(description="Whether the player has made the correct guess about the entity.")


class SearchTool(BaseTool):
    """This can be replaced by any real search tool.

    As we don't have a funding for search API, we use an LLM
    to mock the search process.
    """

    name: str = "search"
    description: str = "Search the web for information"
    model: Optional[CrewAILLM] = Field(default_factory=lambda: CrewAILLM(model="openai/gpt-4.1"))

    async def _run(self, query: str = "") -> str:
        """Asynchronously run the tool"""
        # Your async implementation here
        return f"Processed {query} asynchronously"


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
    guess_llm: Optional[CrewAILLM] = None
    answer_llm: Optional[CrewAILLM] = None
    search_tool: Optional[SearchTool] = None

    def render_history(self) -> str:
        return "\n\n".join(
            [
                f"Question #{i}: {turn.question}\nResponse #{i}: {'yes' if turn.response else 'no'}"
                for i, turn in enumerate(self.interactions, start=1)
            ]
        )


class TwentyQuestionsFlow(Flow[TwentyQuestionsGameState]):

    @start("player")
    async def player(self):
        print("Starting flow")
        agent = CrewAgent(
            role="Player in a game of 20 questions.",
            goal="The answerer has chosen an entity. It can be but not limited to a person, character, place, thing, or concept. Ask a series of yes/no questions to the answerer. Win the game by guessing the entity in 20 questions or less.",
            backstory="Specialized in reasoning and making guesses based on previous guesses and the answerer's responses.",
            tools=[self.state.search_tool] if self.state.search_tool else [],
            llm=self.state.guess_llm,
        )
        query = (
            "Here is a list of questions and responses from previous rounds. Use this information to ask your next question.\n\n"
            + self.state.render_history()
            + '\n\nUse the search tool to gather information about what\'s on your mind.\n\nOutput a yes/no question to ask which would be most helpful for you to determine the answer. If you have a good guess in mind, you can ask "Is it <your guess>?" directly.'
        )

        result = await agent.kickoff_async(query)
        console.print(f"[bold green]Player: [/bold green] {result.raw}")
        self.state.next_question = result.raw

    @listen(player)
    async def answerer(self):
        agent = CrewAgent(
            role="Answerer in a game of 20 questions.",
            goal="The answerer has chosen an entity. Answer the player's yes/no questions about this entity at the best efforts.",
            backstory="Very knowledgeable and specialized in answering yes/no questions about entities.",
            llm=self.state.answer_llm,
        )
        query = (
            f"Your entity in mind is: {self.state.answer}. The player has asked you the following question: {self.state.next_question}\n\n"
            + "Answer yes or no to the question. If the user has made the correct guess and you are asked if it is the correct answer, answer yes."
        )

        result = await agent.kickoff_async(query)
        answerer_response = cast(AnswererResponse, result.pydantic)
        console.print(f"[bold blue]Answerer: [/bold blue] {answerer_response}")
        self.state.interactions.append(Turn(question=self.state.next_question, response=answerer_response.yes_or_no))
        self.state.next_question = ""  # Reset the next question

        if answerer_response.correct:
            self.state.correct = True
        else:
            self.state.correct = False
        print(f"Correct: {self.state.correct}")

    @router(answerer)
    def game_should_continue(self):
        if self.state.correct:
            console.print(f"[bold green]Correct! You win![/bold green]")
            return "game_over"
        elif self.state.turn_index >= 20:
            console.print(
                f"[bold red]You've asked 20 questions and still haven't guessed the entity. You lose![/bold red]"
            )
            return "game_over"
        else:
            self.state.turn_index += 1
            return "player"

    @listen("game_over")
    def finish(self):
        if self.state.correct:
            print("You win!")
        else:
            print("You lose!")


flow = TwentyQuestionsFlow()
flow.plot()
result = flow.kickoff(
    {
        "answer": "c",
        "category": "letter",
    }
)

print(f"Generated fun fact: {result}")
