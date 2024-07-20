from micro.assistant import Assistant
from micro.llm.ollama import Ollama
from micro.tools.duckduckgo import DuckDuckGo

hermes = Assistant(
    llm=Ollama(model="openhermes"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
)
hermes.print_response("Whats happening in France? Summarize top stories with sources.", markdown=True)
