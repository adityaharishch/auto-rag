from micro.assistant import Assistant
from micro.tools.yfinance import YFinanceTools
from micro.llm.anthropic import Claude

assistant = Assistant(
    name="Finance Assistant",
    llm=Claude(model="claude-3-haiku-20240307"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    # debug_mode=True,
)
assistant.print_response("Share the NVDA stock price and analyst recommendations", markdown=True)
# assistant.print_response("Summarize fundamentals for TSLA", markdown=True)
