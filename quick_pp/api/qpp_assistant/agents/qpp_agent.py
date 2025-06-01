from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from quick_pp.api.qpp_assistant.agents.plan_execute_agent import PlanExecuteAgent
from quick_pp.api.qpp_assistant.agents.base_agent import BaseQPPAgent


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


class QPPAgent:
    def __init__(self, llm):
        base_agent = BaseQPPAgent(llm)
        self.qa_agent = PlanExecuteAgent(llm=llm, base_agent=base_agent)
        self.base_agent = base_agent

    def build(self):
        """Build the plan execute agent workflow."""

        workflow = StateGraph(MessagesState)
        # self.workflow.add_node('base_agent', self.base_agent.get_agent_executor())
        workflow.add_node('qa_agent', self.qa_agent.build())
        workflow.add_edge(START, "researcher")

        return workflow.compile()
