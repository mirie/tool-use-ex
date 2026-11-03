# CrewAI tool use example

A single-agent [CrewAI](https://docs.crewai.com) crew that answers "what is the stock price for X?" by calling a custom tool instead of guessing.

Based on the tool use chapter of *Agentic Design Patterns* by Antonio Gulli.

## What it does

1. `main.py` loops over a list of tickers and kicks off the crew once per ticker with an input like `{"ticker": "MSFT"}`.
2. The **Financial Analyst** agent receives the task and calls the `Lookup Stock Price` tool.
3. The tool returns a simulated price, or raises `ValueError` if the ticker isn't in its hardcoded list.
4. The agent reports the price in one sentence, or says clearly that it couldn't be retrieved.

Simulated tickers: `AAPL`, `GOOGL`, `MSFT`, `AMZN`, `NVDA`.

## Layout

This follows the standard CrewAI project layout (what `crewai create crew` generates):

```
crew_ai/
  main.py              # Entry point: env + logging setup, inputs, kickoff()
  crew.py              # @CrewBase class FinanceCrew: wires agents, tasks, tools
  config/
    agents.yaml        # Agent prompts: role, goal, backstory, llm
    tasks.yaml         # Task prompts: description, expected_output, agent
  tools/
    stock_price.py     # get_stock_price tool (@tool "Lookup Stock Price")
```

- **YAML** holds the prompts, so you can tune wording without touching Python.
- **`crew.py`** says how things fit together: which agent gets which tools, task order, process.
- **`main.py`** says how it's run: environment, logging, inputs.

`{ticker}` placeholders in `tasks.yaml` are filled from the `inputs` passed to `kickoff()`.

## Running

From the `tool_use` project root:

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Run either of:

   ```
   uv run crewai-stock-price
   uv run python -m tool_use.examples.crew_ai.main
   ```

The tickers to run are listed in `run()` in `main.py`: `MSFT`, `AMZN`, `NVDA`, plus `TSLA`, which isn't in the simulated prices, to show how the agent handles a tool error.

## Reading the output

- **CrewAI's trace** (`verbose=True` on the agent and crew) is printed directly to the console. It shows the agent's reasoning, each tool call with its input and result, and the final answer.
- **Python log lines** (`INFO` / `WARNING`) come from `main.py` and the tool. The tool logs every call, so you can confirm it actually ran rather than the model answering from memory.

## Notes

- **Model:** set by `llm:` in `config/agents.yaml` (currently `gpt-5-mini`).
- **Tool description:** the tool's docstring is what the model reads when deciding whether to use it, so keep it specific about when to call the tool and what it returns.
- **Tool errors:** a `ValueError` raised in the tool doesn't crash the run. CrewAI passes the error message back to the agent as the tool's result.
