# CrewAI tool use example

A single-agent [CrewAI](https://docs.crewai.com) crew that answers free-form finance questions, deciding for itself which tools to call, if any.

Based on the tool use chapter of *Agentic Design Patterns* by Antonio Gulli.

## What it does

1. `main.py` loops over a list of questions and kicks off the crew once per question with an input like `{"question": "What is the stock price of MSFT?"}`.
2. The **Financial Analyst** agent reads the question and decides which of its tools to call. It may call one tool, several, or none.
3. It answers in one to three sentences, or says clearly when a tool couldn't provide the data.

## Tools

| Tool | File | What it returns |
|---|---|---|
| `Lookup Stock Price` | `tools/stock_price.py` | A simulated price for `AAPL`, `GOOGL`, `MSFT`, `AMZN` or `NVDA`. Any other ticker raises `ValueError`. |
| `Get Current Time` | `tools/market_time.py` | The current time in an IANA timezone such as `America/New_York`. |
| `Get Portfolio Holdings` | `tools/portfolio.py` | The user's private share counts: `MSFT` 12, `NVDA` 40, `AAPL` 25. |

## The questions

Each question in `run()` in `main.py` tests a different tool use behaviour:

| Question | What it shows |
|---|---|
| What is the stock price of MSFT? | One tool, one call |
| What is the stock price of TSLA? | A tool error, which the agent reports instead of guessing |
| Compare the stock prices of NVDA and AMZN. | Several calls to the same tool |
| Is the New York Stock Exchange open right now? | Picking the time tool, not the price tool |
| How much is my NVDA position worth? | Private data plus combining two tools: holdings × price |
| What is a stock ticker symbol? | No tool needed: the agent answers directly |

## Layout

This follows the standard CrewAI project layout (what `crewai create crew` generates):

```
crewai_ex/
  main.py              # Entry point: env + logging setup, questions, kickoff()
  crew.py              # @CrewBase class FinanceCrew: wires agents, tasks, tools
  config/
    agents.yaml        # Agent prompts: role, goal, backstory, llm
    tasks.yaml         # Task prompts: description, expected_output, agent
  tools/
    stock_price.py     # get_stock_price tool (@tool "Lookup Stock Price")
    market_time.py     # get_current_time tool (@tool "Get Current Time")
    portfolio.py       # get_portfolio_holdings tool (@tool "Get Portfolio Holdings")
```

- **YAML** holds the prompts, so you can tune wording without touching Python.
- **`crew.py`** says how things fit together: which agent gets which tools, task order, process.
- **`main.py`** says how it's run: environment, logging, inputs.

The `{question}` placeholder in `tasks.yaml` is filled from the `inputs` passed to `kickoff()`.

## Running

From the `tool_use` project root:

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Run either of:

   ```
   uv run crewai-tool-use
   uv run python -m tool_use.examples.crewai_ex.main
   ```

## Reading the output

- **CrewAI's trace** (`verbose=True` on the agent and crew) is printed directly to the console. It shows the agent's reasoning, each tool call with its input and result, and the final answer.
- **Python log lines** (`INFO` / `WARNING`) come from `main.py` and the tools. Every tool logs its calls, so you can confirm it actually ran rather than the model answering from memory.

## Notes

- **Model:** set by `llm:` in `config/agents.yaml` (currently `gpt-5-mini`).
- **Tool description:** each tool's docstring is what the model reads when deciding whether to use it, so keep it specific about when to call the tool and what it returns.
- **Tool errors:** a `ValueError` raised in a tool doesn't crash the run. CrewAI passes the error message back to the agent as the tool's result.
