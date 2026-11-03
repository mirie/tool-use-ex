"""LangChain tool exploration example"""

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    # TODO: define tools, build the agent, and invoke it here
    logger.info("LangChain example starting")


if __name__ == "__main__":
    run()
