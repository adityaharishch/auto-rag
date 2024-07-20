from micro.assistant import Assistant
from micro.tools.duckduckgo import DuckDuckGo
from micro.llm.ollama import OllamaTools


assistant = Assistant(
    llm=OllamaTools(model="llama3"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
)

assistant.print_response("Whats happening in the US?", markdown=True)
