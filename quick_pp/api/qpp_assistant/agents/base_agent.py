from langchain_ollama.chat_models import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing import Annotated, Dict, Any, List
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver

from quick_pp.api.qpp_assistant.prompt_templates.base_prompt import BASE_PROMPT
from quick_pp.api.qpp_assistant.tools import TOOLS
from quick_pp.logger import logger


class State(BaseModel):
    messages: Annotated[List[Dict[str, str]], add_messages] = Field(default_factory=list)
    input: str = ""
    generated_response: str = ""
    error: str = ""


class BaseAgent:
    def __init__(self, llm: ChatOllama, memory: InMemorySaver, thread_id: str):
        self.llm = llm
        self.thread_id = thread_id
        self.memory = memory
        self.agent_executor = create_react_agent(llm, TOOLS, prompt=BASE_PROMPT, checkpointer=memory)
        logger.info("BaseAgent initialized with tools: %s", [tool.name for tool in TOOLS])

    def setup(self, state: State) -> Dict[str, Any]:
        """Process the input and generate a response using the agent executor."""
        try:
            question = state.input
            if not question.strip():
                return {
                    "input": question,
                    "generated_response": "Please provide a valid question or request.",
                    "error": ""
                }

            config = {"configurable": {"thread_id": self.thread_id}}

            # Prepare the input for the agent
            agent_input = {"messages": [("user", question)]}
            logger.info("Invoking agent executor with input: %s", agent_input)

            # Invoke the agent executor
            raw_response = self.agent_executor.invoke(agent_input, config=config)

            if not raw_response or "messages" not in raw_response:
                raise ValueError("Invalid response format from agent executor")

            response_content = raw_response["messages"][-1].content

            logger.info("Generated response: %s", response_content[:100] + "..."
                        if len(response_content) > 100 else response_content)

            return {
                "input": question,
                "generated_response": response_content,
                "error": ""
            }

        except Exception as e:
            error_msg = f"Error in BaseAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "input": state.input,
                "generated_response": "I apologize, but I encountered an error processing your request.",
                "error": error_msg
            }

    def get_agent_executor(self):
        """Get the agent executor."""
        return self.agent_executor

    def get_thread_id(self):
        """Get the thread id."""
        return self.thread_id

    def get_memory(self):
        """Get the memory."""
        return self.memory
