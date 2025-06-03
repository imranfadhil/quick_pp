import operator
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict

from quick_pp.api.qpp_assistant.prompt_templates.plan_execute_prompt import PLANNER_PROMPT, REPLANNER_PROMPT
from quick_pp.api.qpp_assistant.agents.base_agent import BaseAgent
from quick_pp.logger import logger


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="Different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


class PlanExecuteAgent:

    def __init__(self, llm: ChatOllama):
        """Initialize the PlanExecuteAgent with an LLM and a base agent."""
        self.llm = llm
        self.agent_executor = BaseAgent(llm).get_agent_executor()
        logger.info("PlanExecuteAgent initialized.")

    def execute_step(self, state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:\n{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        logger.debug(f"Executing step: {task}")
        agent_response = self.agent_executor.invoke(
            {"messages": [("user", task_formatted)]}
        )
        logger.debug(f"Agent response: {agent_response['messages'][-1].content}")
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    def plan_step(self, state: PlanExecute):
        logger.info("Generating plan for input.")
        planner = PLANNER_PROMPT | self.llm.with_structured_output(Plan)
        plan = planner.invoke({"messages": [("user", state["input"])]})
        logger.debug(f"Generated plan: {plan.steps}")
        return {"plan": plan.steps}

    def replan_step(self, state: PlanExecute):
        logger.info("Replanning based on current state.")
        replanner = REPLANNER_PROMPT | self.llm.with_structured_output(Act)
        output = replanner.invoke(state)
        if isinstance(output.action, Response):
            logger.debug(f"Returning response: {output.action.response}")
            return {"response": output.action.response}
        else:
            logger.debug(f"Replanned steps: {output.action.steps}")
            return {"plan": output.action.steps}

    def should_end(self, state: PlanExecute):
        if "response" in state and state["response"]:
            logger.info("Workflow ending with response.")
            return END
        else:
            return "agent"

    def build(self):
        logger.info("Building PlanExecuteAgent workflow.")
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("agent", self.execute_step)
        workflow.add_node("replan", self.replan_step)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")

        workflow.add_conditional_edges(
            "replan",
            self.should_end,
            ["agent", END],
        )
        return workflow.compile()
