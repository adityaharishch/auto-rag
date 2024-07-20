from micro.assistant import Assistant
from micro.llm.openai.like import OpenAILike

assistant = Assistant(
    llm=OpenAILike(base_url="http://localhost:8000/v1"),
    description="You help people with their health and fitness goals.",
)
assistant.print_response("Share a quick healthy breakfast recipe.", stream=False, markdown=True)
