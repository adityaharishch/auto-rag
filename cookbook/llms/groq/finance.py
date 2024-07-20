from micro.assistant import Assistant
from micro.tools.yfinance import YFinanceTools
from micro.llm.groq import Groq

assistant = Assistant(
    llm=Groq(model="llama3-70b-8192"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    show_tool_calls=True,
)
assistant.print_response("What's the NVDA stock price", markdown=True)
assistant.print_response("Share NVDA analyst recommendations", markdown=True)
assistant.print_response("Summarize fundamentals for TSLA", markdown=True)
