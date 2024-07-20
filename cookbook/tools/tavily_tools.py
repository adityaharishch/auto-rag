from micro.assistant import Assistant
from micro.tools.tavily import TavilyTools

assistant = Assistant(tools=[TavilyTools()], show_tool_calls=True)
assistant.print_response("Search tavily for 'language models'", markdown=True)
