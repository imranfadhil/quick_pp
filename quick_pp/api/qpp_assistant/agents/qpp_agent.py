from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from quick_pp.api.qpp_assistant.agents.plan_execute_agent import PlanExecuteAgent
from quick_pp.api.qpp_assistant.agents.base_agent import BaseAgent, State
from quick_pp.logger import logger


class QPPAgent:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.base_agent = BaseAgent(llm).setup
        self.qa_agent = PlanExecuteAgent(llm).build()

    def build(self):
        """Build the plan execute agent workflow with proper error handling and routing."""

        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node('base_agent', self._safe_execute(self.base_agent))
        workflow.add_node('qa_agent', self._safe_execute(self.qa_agent))

        # Define routing condition
        def needs_qa_agent(state: Dict[str, Any]) -> bool:
            try:
                return state.get("generated_response", "").strip().lower().startswith("plan:")
            except Exception as e:
                logger.error(f"Error in routing condition: {str(e)}", exc_info=True)
                return False

        # Add edges with conditional routing
        workflow.add_edge(START, "base_agent")
        workflow.add_conditional_edges(
            "base_agent",
            needs_qa_agent,
            {
                True: "qa_agent",
                False: END
            }
        )
        workflow.add_edge("qa_agent", END)

        return workflow.compile()

    def _safe_execute(self, func):
        """Wrapper to safely execute agent functions with error handling."""
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Execute the function synchronously
                result = func(state)
                # If the result is a coroutine, we need to handle it differently
                if hasattr(result, '__await__'):
                    import asyncio
                    result = asyncio.run(result)
                return result
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {
                    "messages": [{"role": "assistant", "content": error_msg}],
                    "generated_response": error_msg,
                    "error": error_msg
                }
        return wrapper
