import chainlit as cl
import json
from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama
from mcp import ClientSession

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


@cl.on_chat_start
async def start_chat():
    try:
        cl.user_session.set("chat_messages", [])
        logger.info("Chat session started")
    except Exception as e:
        logger.error(f"Error starting chat: {str(e)}", exc_info=True)


@cl.on_message
async def on_message(message: cl.Message):
    try:
        # Update chat history
        chat_messages = cl.user_session.get("chat_messages", [])
        chat_messages.append({"role": "user", "content": HumanMessage(content=message.content)})
        cl.user_session.set("chat_messages", chat_messages)

        # Show processing message
        await cl.Message(content="Processing your request...").send()

        # Configure the execution
        config = {
            "recursion_limit": 50,
        }

        # Invoke the agent executor
        try:
            for event in graph_executor.stream({"input": message.content}, config=config, stream_mode='updates'):
                response = event
                logger.info(f"Received event: {event}")
        except Exception as e:
            error_msg = f"Error during graph execution: {str(e)}"
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
                response_content = str(response)
        except Exception as e:
            error_msg = f"Error processing response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await cl.Message(content=error_msg).send()
            return

        # Update chat history and send response
        chat_messages.append({"role": "assistant", "content": response_content})
        cl.user_session.set("chat_messages", chat_messages)
        await cl.Message(elements=[cl.Text(content=response_content)]).send()

    except Exception as e:
        error_msg = f"Unexpected error in message handling: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await cl.Message(content=error_msg).send()
