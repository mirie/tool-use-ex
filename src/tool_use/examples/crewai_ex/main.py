"""CrewAI tool exploration example
Source: Agentic Design Patterns by Antonio Gulli
"""

import logging
import os

from dotenv import load_dotenv

from tool_use.examples.crewai_ex.crew import FinanceCrew

logger = logging.getLogger(__name__)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    questions = [
        # One tool, one call
        "What is the stock price of MSFT?",
        # Tool error: TSLA is not in the simulated prices
        "What is the stock price of TSLA?",
        # Two-part question: multiple calls to the same tool
        "Compare the stock prices of NVDA and AMZN.",
        # Picking a different tool: needs the current time, not a price
        "Is the New York Stock Exchange open right now?",
        # Private data + chaining: holdings -> price -> multiply
        "How much is my NVDA position worth?",
        # No tool needed: the agent should answer directly
        "What is a stock ticker symbol?",
    ]
    for question in questions:
        inputs = {"question": question}
        logger.info("Starting crew with inputs %s", inputs)
        # Build a fresh crew per run so no state carries over between questions
        result = FinanceCrew().crew().kickoff(inputs=inputs)
        logger.info("Crew finished. Q: %s | A: %s", question, result)


if __name__ == "__main__":
    run()
