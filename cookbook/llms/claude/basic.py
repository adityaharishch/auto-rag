from micro.assistant import Assistant
from micro.llm.anthropic import Claude

assistant = Assistant(
    llm=Claude(model="claude-3-haiku-20240307"),
    description="You help people with their health and fitness goals.",
)
assistant.print_response("Share a quick healthy breakfast recipe.", markdown=True)
