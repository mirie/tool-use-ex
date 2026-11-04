# Tool use examples

Examples of the **Tool Use** pattern from *Agentic Design Patterns* by Antonio Gulli, built in different agent frameworks: an agent decides when to call a tool, which one, and what to do with the result.

| Example | Framework | Run |
|---|---|---|
| [LangChain](src/tool_use/examples/langchain_ex/README.md) | LangChain 1.x `create_agent`, with OpenAI, local Ollama (Qwen3) or Fireworks (Kimi K3) models | `uv run langchain-tool-use [--provider openai\|ollama\|fireworks]` |
| [CrewAI](src/tool_use/examples/crewai_ex/README.md) | CrewAI crew with YAML agents and tasks | `uv run crewai-tool-use` |

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python and the dependencies)
- An API key for each hosted provider you use: `OPENAI_API_KEY` (OpenAI, also used by CrewAI) and `FIREWORKS_API_KEY` (Fireworks)
- Optional, for local models: [Ollama](https://ollama.com) and about 10 GB of free disk and memory for `qwen3:14b`

## Setup

1. Install dependencies:

   ```
   uv sync
   ```

2. Copy `.env.example` to `.env` and fill in the keys you need.
3. Optional, for `--provider ollama`:

   ```
   brew install ollama        # or download the app from ollama.com
   ollama serve               # starts the local server on localhost:11434 (the app does this for you)
   ollama pull qwen3:14b      # one-time model download (~9 GB)
   ollama list                # check that the model is there
   ```

## Run

```
uv run langchain-tool-use                        # OpenAI gpt-5-mini
uv run langchain-tool-use --provider ollama      # local qwen3:14b
uv run langchain-tool-use --provider fireworks   # Kimi K3 on Fireworks
uv run crewai-tool-use                           # CrewAI example
```
