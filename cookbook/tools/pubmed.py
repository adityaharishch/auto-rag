from micro.assistant import Assistant
from micro.tools.pubmed import PubmedTools

assistant = Assistant(tools=[PubmedTools()], debug_mode=True, show_tool_calls=True)

assistant.print_response(
    "ulcerative colitis.",
    markdown=True,
)
