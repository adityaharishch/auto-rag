from micro.assistant import Assistant
from micro.tools.yfinance import YFinanceTools
from micro.llm.openai import OpenAIChat

assistant = Assistant(
    name="Finance Assistant",
    llm=OpenAIChat(model="gpt-4-turbo"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
)
assistant.print_response("Share the NVDA stock price and analyst recommendations", markdown=True)
