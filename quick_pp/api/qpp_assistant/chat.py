import chainlit as cl
from chainlit.input_widget import Select
from chainlit.types import ThreadDict
import json
from langchain_ollama.chat_models import ChatOllama
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from mcp import ClientSession
import re
from typing import Dict, Any, List

from quick_pp.api.qpp_assistant.agents.qpp_agent import QPPAgent
from quick_pp.logger import logger


# Initialize the LLM and agent
llm = ChatOllama(model="qwen3", temperature=0.0)
graph_executor = QPPAgent(llm).build()


class ChainlitChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in Chainlit's user session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Any] = []

    @property
    def messages(self) -> List[Any]:
        """Get all messages."""
        return cl.user_session.get("chat_history", [])

    @messages.setter
    def messages(self, value: List[Any]) -> None:
        """Set all messages."""
        cl.user_session.set("chat_history", value)

    def add_user_message(self, message: str) -> None:
        """Add a user message to the store."""
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the store."""
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []


@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    try:
        # List available tools
        result = await session.list_tools()

        # Process tool metadata
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        } for t in result.tools]

        # Store tools for later use
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)

        logger.info(f"Successfully connected to MCP: {connection.name}")
    except Exception as e:
        logger.error(f"Error in MCP connection: {str(e)}", exc_info=True)
        await cl.Message(content=f"Error connecting to MCP: {str(e)}").send()


@cl.step(type="tool")
async def call_tool(tool_use):
    try:
        tool_name = tool_use["name"]
        tool_input = tool_use["input"]

        current_step = cl.context.current_step
        current_step.name = tool_name

        # Identify which mcp is used
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_name = None

        for connection_name, tools in mcp_tools.items():
            if any(tool.get("name") == tool_name for tool in tools):
                mcp_name = connection_name
                break

        if not mcp_name:
            error_msg = f"Tool {tool_name} not found in any MCP connection"
            logger.error(error_msg)
            current_step.output = json.dumps({"error": error_msg})
            return current_step.output

        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

        if not mcp_session:
            error_msg = f"MCP {mcp_name} not found in any MCP connection"
            logger.error(error_msg)
            current_step.output = json.dumps({"error": error_msg})
            return current_step.output

        try:
            current_step.output = await mcp_session.call_tool(tool_name, tool_input)
            logger.info(f"Successfully called tool {tool_name}")
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            current_step.output = json.dumps({"error": error_msg})

        return current_step.output
    except Exception as e:
        error_msg = f"Unexpected error in call_tool: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg})


def setup_runnable() -> None:
    """Setup the runnable chain with message history."""
    session_id = cl.user_session.get("user").identifier
    message_history = ChainlitChatMessageHistory(session_id=session_id)

    runnable = RunnableWithMessageHistory(
        graph_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cl.user_session.set("runnable", runnable)


@cl.password_auth_callback
def auth_callback(username: str):
    return cl.User(identifier=username)


@cl.on_chat_start
async def start_chat():
    try:
        app_user = cl.user_session.get("user")
        await cl.Message(f"Hello {app_user.identifier}").send()

        setup_runnable()

        settings = await cl.ChatSettings(
            [
                Select(
                    id="Show Thinking Process",
                    label="Show Thinking Process",
                    values=["Yes", "No"],
                    initial_index=1,
                )
            ]
        ).send()
        cl.user_session.set("show_thinking_process", settings["Show Thinking Process"] == "Yes")

        logger.info("Chat session started")
    except Exception as e:
        logger.error(f"Error starting chat: {str(e)}", exc_info=True)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    message_history = ChainlitChatMessageHistory(session_id=cl.user_session.get("user").identifier)
    root_messages = [m for m in thread["steps"]]

    for message in root_messages:
        if message["type"] == "user_message":
            message_history.add_user_message(message["output"])
        elif message["type"] == "assistant_message":
            message_history.add_ai_message(message["output"])

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    try:
        runnable = cl.user_session.get("runnable")
        message_history = ChainlitChatMessageHistory(session_id=cl.user_session.get("user").identifier)

        # Configure the execution
        config = RunnableConfig(
            recursion_limit=50,
            configurable={"session_id": cl.user_session.get("user").identifier}
        )

        # Run the chain with message history
        try:
            chain_input: Dict[str, Any] = {"input": message.content}
            response = await runnable.ainvoke(chain_input, config=config)
            logger.info(f"Received response: {response}")
        except Exception as e:
            error_msg = f"Error during chain execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await cl.Message(content=error_msg).send()
            return

        # Process the response
        if not response:
            await cl.Message(content="No response generated from the model.").send()
            return

        try:
            if hasattr(response, "messages") and response["messages"]:
                response_content = response["messages"][-1].content
            else:
                response_content = response["generated_response"]
                if not cl.user_session.get("show_thinking_process"):
                    # Remove thinking process content and tags if user doesn't want to see them
                    response_content = re.sub(r'<think>(.|\n)*?</think>', '', response_content)
        except Exception as e:
            error_msg = f"Error processing response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await cl.Message(content=error_msg).send()
            return

        # Add user message and assistant response to history
        message_history.add_user_message(message.content)
        message_history.add_ai_message(response_content)

        # Send response
        await cl.Message(content=response_content).send()

    except Exception as e:
        error_msg = f"Unexpected error in message handling: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await cl.Message(content=error_msg).send()
