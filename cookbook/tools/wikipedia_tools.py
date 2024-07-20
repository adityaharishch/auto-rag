from micro.assistant import Assistant
from micro.tools.wikipedia import WikipediaTools

assistant = Assistant(tools=[WikipediaTools()], show_tool_calls=True)
assistant.print_response("Search wikipedia for 'ai'", markdown=True)
