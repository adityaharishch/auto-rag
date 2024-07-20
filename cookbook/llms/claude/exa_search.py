from micro.assistant import Assistant
from micro.tools.exa import ExaTools
from micro.tools.website import WebsiteTools
from micro.llm.anthropic import Claude

assistant = Assistant(llm=Claude(), tools=[ExaTools(), WebsiteTools()], show_tool_calls=True)
assistant.print_response(
    "Produce this table: research chromatic homotopy theory."
    "Access each link in the result outputting the summary for that article, its link, and keywords; "
    "After the table output make conceptual ascii art of the overarching themes and constructions",
    markdown=True,
)
