import chainlit as cl
from chainlit.input_widget import Select
import re
import json
from langchain_ollama.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from mcp import ClientSession
from operator import itemgetter
from typing import Dict, Any

from quick_pp.api.qpp_assistant.agents.qpp_agent import QPPAgent
from quick_pp.logger import logger


# Initialize the LLM and agent
llm = ChatOllama(model="qwen3", temperature=0.0)
graph_executor = QPPAgent(llm).build()


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
    """Setup the runnable chain with memory."""
    memory = cl.user_session.get("memory")
    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | graph_executor  # Chain the graph executor after memory
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

        cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
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
async def on_chat_resume(thread):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    try:
        memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
        runnable = cl.user_session.get("runnable")  # type: RunnablePassthrough

        # Show processing message
        await cl.Message(content="Processing your request...").send()

        # Add user message to memory before processing
        memory.chat_memory.add_user_message(message.content)

        # Configure the execution
        config = RunnableConfig(
            recursion_limit=50,
            configurable={"memory": memory}
        )

        # Run the chain with memory
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

        # Add assistant response to memory
        memory.chat_memory.add_ai_message(response_content)

        # Send response
        await cl.Message(content=response_content).send()

    except Exception as e:
        error_msg = f"Unexpected error in message handling: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await cl.Message(content=error_msg).send()
