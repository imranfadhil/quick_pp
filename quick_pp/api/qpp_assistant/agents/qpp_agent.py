from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END

from quick_pp.api.qpp_assistant.agents.plan_execute_agent import PlanExecuteAgent
from quick_pp.api.qpp_assistant.agents.base_agent import BaseAgent, State


class QPPAgent:
    def __init__(self, llm: ChatOllama):
        self.base_agent = BaseAgent(llm).setup
        self.qa_agent = PlanExecuteAgent(llm).build()

    def build(self):
        """Build the plan execute agent workflow."""

        workflow = StateGraph(State)
        workflow.add_node('base_agent', self.base_agent)
        workflow.add_node('qa_agent', self.qa_agent)

        workflow.add_edge(START, "base_agent")
        # workflow.add_edge("base_agent", "qa_agent")
        # workflow.add_edge("qa_agent", END)
        workflow.add_edge("base_agent", END)

        return workflow.compile()
