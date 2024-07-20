from os import getenv

import vertexai
from micro.assistant import Assistant
from micro.llm.gemini import Gemini
from micro.tools.duckduckgo import DuckDuckGo

# *********** Initialize VertexAI ***********
vertexai.init(project=getenv("PROJECT_ID"), location=getenv("LOCATION"))

assistant = Assistant(
    llm=Gemini(model="gemini-pro"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
)
assistant.print_response("Whats happening in France?  Summarize top 10 stories with sources", markdown=True)
