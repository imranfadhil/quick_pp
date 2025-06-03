from langchain_core.messages import BaseMessage
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

from quick_pp.api.qpp_assistant.agents.plan_execute_agent import PlanExecuteAgent
from quick_pp.api.qpp_assistant.agents.base_agent import BaseAgent


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


class QPPAgent:
    def __init__(self, llm: ChatOllama):
        self.base_agent = BaseAgent(llm).get_agent_executor()
        self.qa_agent = PlanExecuteAgent(llm).build()

    def build(self):
        """Build the plan execute agent workflow."""

        workflow = StateGraph(State)
        workflow.add_node('base_agent', self.base_agent)
        workflow.add_node('qa_agent', self.qa_agent)

        workflow.add_edge(START, "base_agent")
        workflow.add_edge("base_agent", END)

        return workflow.compile()
