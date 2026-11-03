from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from tool_use.examples.crew_ai.tools.stock_price import get_stock_price


@CrewBase
class FinanceCrew:
    """Single-agent crew that looks up stock prices with a tool"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],  # type: ignore[index]
            tools=[get_stock_price],
            verbose=True,
            allow_delegation=False,  # No need to delegate to other agents for this simple task
        )

    @task
    def stock_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["stock_analysis_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,  # CrewAI prints agent reasoning, tool calls and results to the console
        )
