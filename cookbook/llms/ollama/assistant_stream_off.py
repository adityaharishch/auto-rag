from micro.assistant import Assistant
from micro.llm.ollama import Ollama

assistant = Assistant(
    llm=Ollama(model="llama3"),
    description="You help people with their health and fitness goals.",
)
assistant.print_response("Share a quick healthy breakfast recipe.", stream=False, markdown=True)
