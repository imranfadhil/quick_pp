from langgraph.prebuilt import create_react_agent

from quick_pp.api.qpp_assistant.prompt_templates.base_prompt import BASE_PROMPT
from quick_pp.api.qpp_assistant import TOOLS


class BaseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.agent_executor = create_react_agent(llm, TOOLS, prompt=BASE_PROMPT)

    def get_agent_executor(self):
        """Get the agent executor."""
        return self.agent_executor
