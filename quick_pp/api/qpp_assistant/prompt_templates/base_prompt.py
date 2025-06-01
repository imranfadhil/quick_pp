from langchain_core.prompts import ChatPromptTemplate

BASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful Petrophysics speacialist ready to coach. "
                "Your task is to assist the user in achieving their objective by providing step-by-step guidance. "
                "Make sure to provide clear and concise instructions for each step."
            ),
        ),
        ("placeholder", "{messages}"),
    ]
)
