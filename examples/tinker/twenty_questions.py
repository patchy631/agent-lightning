from typing import TypedDict

from crewai import LLM as CrewAILLM
from crewai.flow import Flow, listen, router, start
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel

llm = CrewAILLM(model="openai/gpt-4o-mini")


class TwentyQuestionsGameState(BaseModel):
    answer: str = ""
    category: str = ""
    guess: str = ""
    correct: bool = False
    turn_index: int = 1


class SearchTool(BaseTool):
    """This can be replaced by any real search tool.

    As we don't have a funding for search API, we use an LLM
    to mock the search process.
    """

    name: str = "search"
    description: str = "Search the web for information"

    async def _run(self, query: str = "") -> str:
        """Asynchronously run the tool"""
        # Your async implementation here
        await asyncio.sleep(1)
        return f"Processed {query} asynchronously"


class TwentyQuestionsFlow(Flow[TwentyQuestionsGameState]):

    @start("guess")
    def generate_guess(self):
        print("Starting flow")
        import random

        self.state.guess = random.choice(["a", "b", "c", "d", "e"])
        print(f"Guess: {self.state.guess}")

    @listen(generate_guess)
    def check_guess(self):
        if self.state.guess == self.state.answer:
            self.state.correct = True
        else:
            self.state.correct = False
        print(f"Correct: {self.state.correct}")

    @router(check_guess)
    def game_should_continue(self):
        if self.state.correct or self.state.turn_index >= 20:
            return "game_over"
        else:
            self.state.turn_index += 1
            return "guess"

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
