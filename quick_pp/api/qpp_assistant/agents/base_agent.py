from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing import Annotated
from pydantic import BaseModel

from quick_pp.api.qpp_assistant.prompt_templates.base_prompt import BASE_PROMPT
from quick_pp.api.qpp_assistant import TOOLS


class State(BaseModel):
    messages: Annotated[list[dict], add_messages] = []
    input: str = ""
    generated_response: str = ""


class BaseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.agent_executor = create_react_agent(llm, TOOLS, prompt=BASE_PROMPT)

    def setup(self, state: State) -> dict:

        question = state.input
        history = state.messages[-10:]

        # Invoke the agent executor
        raw_response = self.agent_executor.invoke({"messages": history + [{"role": "user", "content": question}]})
        response_content = raw_response["messages"][-1].content

        return {
            "input": question,
            "generated_response": response_content
        }

    def get_agent_executor(self):
        """Get the agent executor."""
        return self.agent_executor
