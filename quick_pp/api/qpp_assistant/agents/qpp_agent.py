from langchain_ollama.chat_models import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from quick_pp.api.qpp_assistant.agents.plan_execute_agent import PlanExecuteAgent
from quick_pp.api.qpp_assistant.agents.base_agent import BaseAgent, State
from quick_pp.logger import logger


class QPPAgent:
    def __init__(self, llm: ChatOllama, memory: InMemorySaver, thread_id: str):
        self.llm = llm
        self.base_agent = BaseAgent(llm, memory, thread_id)
        self.qa_agent = PlanExecuteAgent(llm, self.base_agent).build()
        self.memory = memory
        self.workflow = None

    def build(self):
        """Build the plan execute agent workflow with proper error handling and routing."""
        if self.workflow is None:
            self.workflow = StateGraph(State)

            # Add nodes
            self.workflow.add_node('base_agent', self.base_agent.setup)
            self.workflow.add_node('qa_agent', self.qa_agent)

            # Define routing condition
            def needs_qa_agent(state: Dict[str, Any]) -> bool:
                try:
                    return state.get("generated_response", "").strip().lower().startswith("plan:")
                except Exception as e:
                    logger.error(f"Error in routing condition: {str(e)}", exc_info=True)
                    return False

            # Add edges with conditional routing
            self.workflow.add_edge(START, "base_agent")
            self.workflow.add_conditional_edges(
                "base_agent",
                needs_qa_agent,
                {
                    True: "qa_agent",
                    False: END
                }
            )
            self.workflow.add_edge("qa_agent", END)

        return self.workflow.compile(checkpointer=self.memory)
