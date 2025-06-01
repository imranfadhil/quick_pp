from langchain_core.prompts import ChatPromptTemplate

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "For the given objective, come up with a simple step by step plan. "
                "This plan should involve individual tasks, that if executed correctly will yield the correct answer. "
                "Do not add any superfluous steps. The result of the final step should be the final answer. "
                "Make sure that each step has all the information needed - do not skip steps."
            ),
        ),
        ("placeholder", "{messages}"),
    ]
)

REPLANNER_PROMPT = ChatPromptTemplate.from_template(
    "For the given objective, come up with a simple step by step plan. "
    "This plan should involve individual tasks, that if executed correctly will yield the correct answer. "
    "Do not add any superfluous steps. The result of the final step should be the final answer. "
    "Make sure that each step has all the information needed - do not skip steps.\n\n"
    "Your objective was this:\n"
    "{input}\n\n"
    "Your original plan was this:\n"
    "{plan}\n\n"
    "You have currently done the follow steps:\n"
    "{past_steps}\n\n"
    "Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. "
    "Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. "
    "Do not return previously done steps as part of the plan."
)

CONCLUDE_PROMPT = ChatPromptTemplate.from_template(
    "Your task is to conclude and generate a clear and concise response to the user based on the plan and past steps. "
    "Your objective was this:\n"
    "{input}\n\n"
    "You have currently done the follow steps:\n"
    "{past_steps}\n\n"
)
