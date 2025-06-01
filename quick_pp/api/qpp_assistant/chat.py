import chainlit as cl
import json
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama.chat_models import ChatOllama
from mcp import ClientSession

from quick_pp.api.qpp_assistant.agents.base_agent import BaseQPPAgent
from quick_pp.api.qpp_assistant.agents.plan_execute_agent import PlanExecuteAgent

llm = ChatOllama(model="granite3.2", temperature=0.0)
graph_executor = PlanExecuteAgent(llm).build()
agent_executor = BaseQPPAgent(llm).get_agent_executor()


@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
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


@cl.step(type="tool")
async def call_tool(tool_use):
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
        current_step.output = json.dumps({"error": f"Tool {tool_name} not found in any MCP connection"})
        return current_step.output

    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

    if not mcp_session:
        current_step.output = json.dumps({"error": f"MCP {mcp_name} not found in any MCP connection"})
        return current_step.output

    try:
        current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    except Exception as e:
        current_step.output = json.dumps({"error": str(e)})

    return current_step.output


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_messages", [])


@cl.on_message
async def on_message(message: cl.Message):

    chat_messages = cl.user_session.get("chat_messages", [])
    chat_messages.append({"role": "user", "content": HumanMessage(content=message.content)})
    cl.user_session.set("chat_messages", chat_messages)

    # Prepare the input for the agent executor
    input_data = {"input": chat_messages}

    # Invoke the agent executor
    response = graph_executor.invoke(
        input_data,
        config=RunnableConfig(
            callbacks=[
                cl.LangchainCallbackHandler(
                    to_ignore=[
                        "ChannelRead",
                        "RunnableLambda",
                        "ChannelWrite",
                        "__start__",
                        "_execute",
                    ]
                )
            ]
        )
    )
    response_content = response["messages"][-1].content if response["messages"] else "No response generated."
    chat_messages.append({"role": "assistant", "content": response_content})
    cl.user_session.set("chat_messages", chat_messages)
    await cl.Message(content=response_content).send()
