from mcp import ClientSession
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
import re
import json


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


async def call_ollama(messages):

    # Prepare tool descriptions for the prompt
    mcp_tools = cl.user_session.get("mcp_tools", {})
    tool_names = ", ".join(
        tool['name']
        for tool_list in mcp_tools.values() for tool in tool_list
    )
    tool_names = tool_names.replace('{', '{{').replace('}', '}}')
    tool_descriptions = "\n".join(
        f"- {tool['name']}: {tool['description']}"
        for tool_list in mcp_tools.values() for tool in tool_list
    )
    tool_descriptions = tool_descriptions.replace('{', '{{').replace('}', '}}')

    # Create the prompt template with tools info
    prompt = ChatPromptTemplate.from_messages([
        ("system", f'''
You are a Petrophysics specialist and you need to answer user questions accurately and efficiently.
You can call tools to answer user questions.
You can use the following tools: {tool_names}.
Here are the tools' descriptions: {tool_descriptions}.
When a user asks a question, select the single best tool to answer it (if any).
If a tool is relevant, respond ONLY in the following JSON format:
{{{{"tool_call": {{{{"name": "<tool_name>", "input": {{{{<input_parameters>}}}}}}}}}}}}
If no tool is relevant, answer the question directly in natural language.
Do not explain your reasoning. Do not output anything except the JSON or the direct answer.
If you need more information from the user to use a tool, ask for it directly.
Example: If the user asks 'Calculate porosity for well X', and there is a 'calculate_porosity' tool, respond with:
{{{{"tool_call": {{{{"name": "calculate_porosity", "input": {{{{"well_id": "X"}}}}}}}}}}}}
        '''),
        ("user", "Please answer the question and use the tools if needed."),
        ("user", "{input}")
    ])

    llm = ChatOllama(model='qwen3', num_ctx=8192, temperature=0.9)
    chain = prompt | llm
    user_input = messages[-1]["content"] if messages else ""
    response = chain.invoke({
        "input": user_input,
    })
    # Remove all substrings bounded by <think>...</think>
    if isinstance(response.content, str):
        response.content = re.sub(r"<think>(.|\n)*?</think>", "", response.content, flags=re.DOTALL)

    return response


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_messages", [])


@cl.on_message
async def on_message(msg: cl.Message):
    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "user", "content": msg.content})
    response = await call_ollama(chat_messages)

    if "tool_call" in response.content:
        # Ensure response.content is a string and extract the JSON containing "tool_call"
        tool_call_dict = None
        if isinstance(response.content, str):
            match = re.search(r'(\{.*"tool_call".*\})', response.content, re.DOTALL)
            if match:
                try:
                    tool_call_dict = json.loads(match.group(1))
                except Exception:
                    tool_call_dict = None
        if tool_call_dict:
            tool_use = tool_call_dict["tool_call"]
            tool_result = await call_tool(tool_use)

            messages = [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_name": tool_use["name"],
                            "content": str(tool_result),
                        }
                    ],
                },
            ]

            chat_messages.extend(messages)

            response = await call_ollama(chat_messages)

    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )

    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "assistant", "content": final_response})

    # Send a response back to the user
    await cl.Message(content=str(response.content)).send()
