# LangChain tool use example

A single [LangChain](https://docs.langchain.com) agent (`create_agent`, LangChain 1.x) that answers free-form questions, deciding for itself which tools to call, if any. The same questions can be run against a frontier model, a local open-weight model, or a hosted open-weight model.

Based on the tool use chapter of *Agentic Design Patterns* by Antonio Gulli.

## What it does

1. `run()` in `main.py` builds one agent with three tools and a short system prompt.
2. It loops over a list of questions and calls `agent.invoke({"messages": [...]})` for each one.
3. The model decides whether to call a tool, which one, and how many times. Each message in the exchange is pretty-printed: question, tool calls, tool results, final answer.

## Tools

| Tool | What it returns |
|---|---|
| `search_information` | A simulated "verified knowledge base": London weather, capital of France, Earth's population, tallest mountain. Keyword match, so rephrased queries still hit. |
| `get_current_time` | The current time in an IANA timezone such as `Europe/Paris`. |
| `search_company_faq` | Private Acme Corp FAQ entries (PTO, remote work, expenses) the model can't know from training. |

The docstring of each tool is what the model reads when deciding whether to use it.

## The questions

| Question | What it shows |
|---|---|
| What is the capital of France? | Whether the tool description persuades the model to search instead of answering from memory |
| What is the weather in London? | Live data: a tool is clearly needed |
| Why do cats sleep so much? | General knowledge: the model may skip the tool |
| Hi! What can you help me with? | No tool needed |
| What's the weather in London and the population of Earth? | Two-part question: multiple, possibly parallel, tool calls |
| What time is it right now in the capital of France? | Chaining: look up the capital, then the time (or skip the lookup) |
| How many PTO days do Acme employees get? | Private data: the tool is the only source |

## Models

Pick a provider with `--provider`. Each one has a default model, and `--model` overrides it.

| `--provider` | Default model | Kind | Needs |
|---|---|---|---|
| `openai` (default) | `gpt-5-mini` | Frontier, hosted | `OPENAI_API_KEY` |
| `ollama` | `qwen3:14b` | Open-weight, local | [Ollama](https://ollama.com) running, model pulled |
| `fireworks` | `accounts/fireworks/models/kimi-k2p6` (Kimi K2.6) | Open-weight, hosted | `FIREWORKS_API_KEY` |

`create_agent` receives a `"provider:model"` string and hands it to `init_chat_model`, which loads the matching integration (`langchain-openai`, `langchain-ollama` or `langchain-fireworks`).

For local models, only use ones trained for tool calling (the **tools** tag on ollama.com). Kimi is too large to run locally (about 1T parameters), so it runs on Fireworks.

## Running

From the `tool_use` project root:

1. Copy `.env.example` to `.env` and fill in the keys for the providers you'll use.
2. For Ollama, once:

   ```
   brew install ollama
   ollama serve            # or open the Ollama app
   ollama pull qwen3:14b
   ```

3. Run:

   ```
   uv run langchain-tool-use                                  # gpt-5-mini
   uv run langchain-tool-use --provider ollama                # qwen3:14b, local
   uv run langchain-tool-use --provider fireworks             # Kimi K2.6, hosted
   uv run langchain-tool-use --provider ollama --model qwen3:30b-a3b
   uv run langchain-tool-use --help
   ```

## Notes

- **Tool call cap:** `ToolCallLimitMiddleware(run_limit=10)` allows at most 10 tool calls per question. Past that, further calls get an error and the model has to answer. This matters more for small models, which can loop.
- **Retries are the model's choice:** LangChain doesn't retry a tool that returns "no result". The model decides whether to search again with a different query, and when to stop.
- **Comparing models:** run the same questions with each provider and compare the traces. Things to watch: unnecessary tool calls, malformed arguments (such as a city name passed as a timezone), parallel vs sequential calls, and whether question 6 chains two tools.
- **Kimi reasoning:** Kimi K2.6 reasons before answering. `ChatFireworks` keeps that reasoning (`reasoning_content`) across tool-call turns, as Kimi expects.
