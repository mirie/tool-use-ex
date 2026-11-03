"""CrewAI tool exploration example
   Source: Agentic Design Patterns by Antonio Gulli
"""

import logging
import os

from dotenv import load_dotenv

from tool_use.examples.crew_ai.crew import FinanceCrew

logger = logging.getLogger(__name__)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    tickers = ["MSFT", "AMZN", "NVDA", "TSLA"]  # TSLA is not in the simulated prices
    for ticker in tickers:
        inputs = {"ticker": ticker}
        logger.info("Starting crew with inputs %s", inputs)
        # Build a fresh crew per run so no state carries over between tickers
        result = FinanceCrew().crew().kickoff(inputs=inputs)
        logger.info("Crew finished for %s. Final result: %s", ticker, result)


if __name__ == "__main__":
    run()
