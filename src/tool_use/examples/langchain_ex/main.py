"""LangChain tool exploration example
Source: Agentic Design Patterns by Antonio Gulli
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MODEL = "gpt-5-mini"


@tool
def search_information(query: str) -> str:
    """
    Search an up-to-date, verified knowledge base. It is more current and reliable
    than your training data, so use it for any factual question (geography, weather,
    science, animals, etc.) before answering. Pass a short search phrase such as
    'capital of France' or 'weather in London'.
    """
    logger.info("Tool called: search_information(query=%r)", query)

    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
    }

    # Keyword match: the model rephrases queries (e.g. "current weather in London"),
    # so look for a known key anywhere in the query instead of an exact match.
    q = query.lower()
    result = next(
        (answer for key, answer in simulated_results.items() if key in q),
        f"Simulated search result for '{query}': No specific information found, "
        "but the topic seems interesting.",
    )

    logger.info("Tool result: %s", result)
    return result


# Experiment 3: a second tool, so the model must choose between tools (or chain them)
@tool
def get_current_time(timezone: str) -> str:
    """
    Get the current local date and time in an IANA timezone such as 'Europe/Paris'
    or 'America/New_York'. Use this whenever a question involves the current time.
    """
    logger.info("Tool called: get_current_time(timezone=%r)", timezone)
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        result = f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/Paris'."
    else:
        result = now.strftime("%A %Y-%m-%d %H:%M %Z")
    logger.info("Tool result: %s", result)
    return result


# Experiment 4: private data the model cannot know from its training
@tool
def search_company_faq(query: str) -> str:
    """
    Search Acme Corp's internal employee FAQ (PTO, remote work, expenses).
    Pass a short search phrase such as 'pto' or 'remote work'.
    """
    logger.info("Tool called: search_company_faq(query=%r)", query)

    faq = {
        "pto": "Acme employees get 22 PTO days per year, plus 3 floating holidays.",
        "remote": "Acme is remote-first; employees come to the office one week per quarter.",
        "expense": "Expenses under $75 need no receipt; anything above needs manager approval.",
    }

    q = query.lower()
    result = next(
        (answer for key, answer in faq.items() if key in q),
        f"No FAQ entry found for '{query}'.",
    )
    logger.info("Tool result: %s", result)
    return result


# --- Create agent with its tools ---
def build_agent():
    return create_agent(
        model=MODEL,
        tools=[search_information, get_current_time, search_company_faq],
        system_prompt=(
            "You are a helpful information assistant. Use your tools when a "
            "question needs information from them; otherwise answer directly. "
            "Keep answers brief."
        ),
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    agent = build_agent()

    logger.info("LangChain example starting")
    questions = [
        "What is the capital of France?",
        "What is the weather in London?",
        "Why do cats sleep so much?",
        "Hi! What can you help me with?",
        # Experiment 2: two-part question -> multiple (possibly parallel) tool calls?
        "What's the weather in London and the population of Earth?",
        # Experiment 3: chaining -> look up the capital, then get the time there
        "What time is it right now in the capital of France?",
        # Experiment 4: private data -> the model has no choice but to use the tool
        "How many PTO days do Acme employees get?",
    ]

    for question in questions:
        result = agent.invoke({"messages": [{"role": "human", "content": question}]})
        # Print the full exchange: user question, tool calls, tool results, final answer
        for message in result["messages"]:
            message.pretty_print()


if __name__ == "__main__":
    run()
