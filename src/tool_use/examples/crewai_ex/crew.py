from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from tool_use.examples.crewai_ex.tools.market_time import get_current_time
from tool_use.examples.crewai_ex.tools.portfolio import get_portfolio_holdings
from tool_use.examples.crewai_ex.tools.stock_price import get_stock_price


@CrewBase
class FinanceCrew:
    """Single-agent crew that answers finance questions, choosing among its tools"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],  # type: ignore[index]
            tools=[get_stock_price, get_current_time, get_portfolio_holdings],
            verbose=True,
            allow_delegation=False,  # No need to delegate to other agents for this simple task
        )

    @task
    def finance_question_task(self) -> Task:
        return Task(
            config=self.tasks_config["finance_question_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,  # CrewAI prints agent reasoning, tool calls and results to the console
        )
