# Tool use examples

Examples of the **Tool Use** pattern from *Agentic Design Patterns* by Antonio Gulli, built in different agent frameworks: an agent decides when to call a tool, which one, and what to do with the result.

| Example | Framework | Run |
|---|---|---|
| [LangChain](src/tool_use/examples/langchain_ex/README.md) | LangChain 1.x `create_agent`, with OpenAI, local Ollama (Qwen3) or Fireworks (Kimi K2.6) models | `uv run langchain-tool-use [--provider openai\|ollama\|fireworks]` |
| [CrewAI](src/tool_use/examples/crewai_ex/README.md) | CrewAI crew with YAML agents and tasks | `uv run crewai-tool-use` |

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync`.
2. Copy `.env.example` to `.env` and set the API keys you need (`OPENAI_API_KEY`, `FIREWORKS_API_KEY`).
