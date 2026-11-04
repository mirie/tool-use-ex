"""LangChain tool exploration example
Source: Agentic Design Patterns by Antonio Gulli
"""

import argparse
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Provider -> (default model in "provider:model" form, required API key env var).
# create_agent passes the string to init_chat_model, which picks the integration.
PROVIDERS = {
    "openai": ("openai:gpt-5-mini", "OPENAI_API_KEY"),
    # Local open-weight model; needs `ollama serve` and `ollama pull qwen3:14b`
    "ollama": ("ollama:qwen3:14b", None),
    # Hosted open-weight model (Kimi K3) on Fireworks, serverless
    "fireworks": ("fireworks:accounts/fireworks/models/kimi-k3", "FIREWORKS_API_KEY"),
}

# Short names for --model, per provider, so you don't have to type long tags.
# Ollama names need a prior `ollama pull <tag>`.
MODEL_ALIASES = {
    "ollama": {
        "qwen3-instruct": "qwen3:30b-a3b-instruct-2507-q4_K_M",  # non-reasoning Qwen3
        "mistral-small": "mistral-small3.2:24b",
        "granite": "granite4:small-h",
        "llama": "llama3.1:8b",
    },
    # Serverless on Fireworks
    "fireworks": {
        "deepseek-flash": "accounts/fireworks/models/deepseek-v4p1-flash-1017",  # no reasoning
        "minimax-flash": "accounts/fireworks/models/minimax-m3p1-flash",  # no reasoning
        "gpt-oss": "accounts/fireworks/models/gpt-oss-120b",
        "nemotron": "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
        "qwen-max": "accounts/fireworks/models/qwen3p8-max",
    },
}


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
def build_agent(model: str, **model_kwargs):
    # init_chat_model resolves "provider:model" and forwards extra kwargs (e.g.
    # reasoning=False for Ollama) to the provider's chat model class.
    return create_agent(
        model=init_chat_model(model, **model_kwargs),
        tools=[search_information, get_current_time, search_company_faq],
        system_prompt=(
            "You are a helpful information assistant. Use your tools when a "
            "question needs information from them; otherwise answer directly. "
            "Keep answers brief."
        ),
        # Hard cap: at most 10 tool calls (across all tools) per question. Past the
        # cap, further calls get an error message and the model must answer.
        middleware=[ToolCallLimitMiddleware(run_limit=10)],
    )


def run() -> None:
    parser = argparse.ArgumentParser(description="LangChain tool use example")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="openai",
        help="Model provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        help="Override the provider's default model: a full name such as 'gpt-4.1' or "
        "'accounts/fireworks/models/gpt-oss-120b', or a short name. ollama: "
        + ", ".join(MODEL_ALIASES["ollama"])
        + "; fireworks: "
        + ", ".join(MODEL_ALIASES["fireworks"]),
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Ollama only: turn off Qwen3's hidden reasoning (about 4x faster per call)",
    )
    args = parser.parse_args()
    if args.no_think and args.provider != "ollama":
        parser.error("--no-think only applies to --provider ollama")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv()
    default_model, key_var = PROVIDERS[args.provider]
    if key_var and not os.getenv(key_var):
        raise SystemExit(f"Set {key_var} in your environment or .env file.")
    model = default_model
    if args.model:
        # Accept the model with or without the "provider:" prefix
        name = args.model.removeprefix(f"{args.provider}:")
        name = MODEL_ALIASES.get(args.provider, {}).get(name, name)
        model = f"{args.provider}:{name}"

    model_kwargs = {"reasoning": False} if args.no_think else {}
    agent = build_agent(model, **model_kwargs)

    logger.info(
        "LangChain example starting with model %s %s", model, model_kwargs or ""
    )
    questions = [
        "What is the capital of France?",
        "What is the weather in London?",
        "Why do cats sleep so much?",
        "Hi! What can you help me with?",
        # Experiment 2: two-part question -> multiple (possibly parallel) tool calls?
        # The tool knows London's weather but has no population entry, so the model
        # must retry, give up, or fall back on its training data for the second part.
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
