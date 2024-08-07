from micro.assistant import Assistant
from micro.llm.mistral import Mistral

assistant = Assistant(
    llm=Mistral(model="mistral-large-latest"),
    description="You help people with their health and fitness goals.",
)
assistant.print_response("Share a quick healthy breakfast recipe.", markdown=True, stream=False)
