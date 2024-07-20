from micro.assistant import Assistant
from micro.llm.openai import OpenAIChat
from micro.tools.duckduckgo import DuckDuckGo

assistant = Assistant(
    llm=OpenAIChat(model="gpt-4-turbo", max_tokens=500, temperature=0.3), tools=[DuckDuckGo()], show_tool_calls=True
)
assistant.print_response("Whats happening in France?", markdown=True)
