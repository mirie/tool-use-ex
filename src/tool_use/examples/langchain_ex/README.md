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
| `search_information` | A simulated "verified knowledge base": London weather, capital of France, tallest mountain. Anything else returns a "no specific information found" message. Keyword match, so rephrased queries still hit. |
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
| What's the weather in London and the population of Earth? | Two-part question, but the tool has no population entry: does the model retry, give up, or answer from its own knowledge? |
| What time is it right now in the capital of France? | Chaining: look up the capital, then the time (or skip the lookup) |
| How many PTO days do Acme employees get? | Private data: the tool is the only source |

## Models

Pick a provider with `--provider`. Each one has a default model, and `--model` overrides it.

| `--provider` | Default model | Kind | Needs |
|---|---|---|---|
| `openai` (default) | `gpt-5-mini` | Frontier, hosted | `OPENAI_API_KEY` |
| `ollama` | `qwen3:14b` | Open-weight, local | [Ollama](https://ollama.com) running, model pulled |
| `fireworks` | `accounts/fireworks/models/kimi-k3` (Kimi K3) | Open-weight, hosted | `FIREWORKS_API_KEY` |

### Models to compare

Beyond the defaults, these are good for comparing reasoning and non-reasoning models. Short names work with `--provider ollama --model <name>`.

| Model | How to run | Reasoning? | Download |
|---|---|---|---|
| Qwen3 14B | `--provider ollama` (default) | Hybrid: thinks by default, `--no-think` turns it off | 9.3 GB |
| Qwen3 30B-A3B 2507 Instruct | `--provider ollama --model qwen3-instruct` | No | 18.6 GB |
| Mistral Small 3.2 (24B) | `--provider ollama --model mistral-small` | No | 15.2 GB |
| IBM Granite 4 small-h (32B MoE) | `--provider ollama --model granite` | No | 19.5 GB |
| Llama 3.1 8B | `--provider ollama --model llama` | No | about 5 GB |
| GPT-4.1 | `--provider openai --model gpt-4.1` | No | hosted |
| DeepSeek V4.1 Flash (1017) | `--provider fireworks --model deepseek-flash` | No (none seen in a test call) | hosted |
| MiniMax M3.1 Flash | `--provider fireworks --model minimax-flash` | No (none seen in a test call) | hosted |
| gpt-oss-120b | `--provider fireworks --model gpt-oss` | Yes | hosted |
| Nemotron Lightning 3.5 (30B-A3B) | `--provider fireworks --model nemotron` | Yes, brief | hosted |
| Qwen3.8 Max | `--provider fireworks --model qwen-max` | Yes | hosted |
| gpt-5-mini | `--provider openai` (default) | Yes | hosted |
| Kimi K3 | `--provider fireworks` (default) | Yes | hosted |

Fireworks only offers models from its serverless list (see `GET https://api.fireworks.ai/inference/v1/models` with your key); the "No" entries were checked with one tool-call test, so a model may still reason on harder questions. Pull the local ones first, e.g. `ollama pull mistral-small3.2:24b` (the exact tags are in `MODEL_ALIASES` in `main.py`). Llama 3.3 is left out because it only comes in 70B (42.5 GB), too big for a 48 GB Mac.

`create_agent` receives a `"provider:model"` string and hands it to `init_chat_model`, which loads the matching integration (`langchain-openai`, `langchain-ollama` or `langchain-fireworks`).

For local models, only use ones trained for tool calling (the **tools** tag on ollama.com). Kimi models are far too large to run locally (K3 is about 2.8T parameters), so Kimi runs on Fireworks. Use a model marked **Serverless** on Fireworks (pay per token); others, such as Kimi K2.6, need a paid dedicated deployment.

## Running

From the `tool_use` project root:

1. Copy `.env.example` to `.env` and fill in the keys for the providers you'll use.
2. For `--provider ollama`, set up Ollama once:

   ```
   brew install ollama     # or download the app from ollama.com
   ollama serve            # local server on localhost:11434; the app starts it for you
   ollama pull qwen3:14b   # ~9 GB download
   ollama list             # confirm the model is installed
   ```

   Ollama must be running whenever you use this provider. If you see a connection error, start `ollama serve`. If you see "model not found", run `ollama pull` for that model. No API key is needed.

3. Run:

   ```
   uv run langchain-tool-use                                  # gpt-5-mini
   uv run langchain-tool-use --provider ollama                # qwen3:14b, local
   uv run langchain-tool-use --provider fireworks             # Kimi K3, hosted
   uv run langchain-tool-use --provider ollama --no-think     # Qwen3 without hidden reasoning, ~4x faster
   uv run langchain-tool-use --provider ollama --model qwen3-instruct
   uv run langchain-tool-use --provider ollama --model mistral-small
   uv run langchain-tool-use --provider openai --model gpt-4.1
   uv run langchain-tool-use --help
   ```

## Notes

- **Latency:** every question takes at least two model calls (ask for the tool, then answer). Qwen3 reasons before each reply, about 100 hidden tokens, which takes 8-9 s per call on an M4 Pro. `--no-think` (Ollama only) turns that off: about 2 s per call. Compare tool choices with and without it.
- **Tool call cap:** `ToolCallLimitMiddleware(run_limit=10)` allows at most 10 tool calls per question. Past that, further calls get an error and the model has to answer. This matters more for small models, which can loop.
- **Retries are the model's choice:** LangChain doesn't retry a tool that returns "no result". The model decides whether to search again with a different query, and when to stop.
- **Comparing models:** run the same questions with each provider and compare the traces. Things to watch: unnecessary tool calls, malformed arguments (such as a city name passed as a timezone), parallel vs sequential calls, and whether question 6 chains two tools.
- **Kimi reasoning:** Kimi reasons before answering. `ChatFireworks` keeps that reasoning (`reasoning_content`) across tool-call turns, as Kimi expects.
