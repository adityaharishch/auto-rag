import json
from pathlib import Path
from typing import Optional
from textwrap import dedent
from typing import List

from phi.assistant import Assistant
from phi.tools import Toolkit
from phi.tools.exa import ExaTools
from phi.tools.shell import ShellTools
from phi.tools.calculator import Calculator
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
# from phi.tools.projectassisstant import ProjectTools
from phi.tools.file import FileTools
from phi.llm.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.embedder.openai import OpenAIEmbedder
from phi.assistant.duckdb import DuckDbAssistant
from phi.assistant.python import PythonAssistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.utils.log import logger
from phi.vectordb.pgvector import PgVector2

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
cwd = Path(__file__).parent.resolve()
scratch_dir = cwd.joinpath("scratch")
if not scratch_dir.exists():
    scratch_dir.mkdir(exist_ok=True, parents=True)


def get_llm_os(
    llm_id: str = "gpt-3.5-turbo",
    calculator: bool = False,
    ddg_search: bool = False,
    file_tools: bool = False,
    shell_tools: bool = False,
    data_analyst: bool = False,
    python_assistant: bool = False,
    research_assistant: bool = False,
    investment_assistant: bool = False,
    project_assistant: bool = False,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    logger.info(f"-*- Creating {llm_id} LLM OS -*-")

    # Add tools available to the LLM OS
    tools: List[Toolkit] = []
    extra_instructions: List[str] = []
    if calculator:
        tools.append(
            Calculator(
                add=True,
                subtract=True,
                multiply=True,
                divide=True,
                exponentiate=True,
                factorial=True,
                is_prime=True,
                square_root=True,
            )
        )
    if ddg_search:
        tools.append(DuckDuckGo(fixed_max_results=3))
    if shell_tools:
        tools.append(ShellTools())
        extra_instructions.append(
            "You can use the `run_shell_command` tool to run shell commands. For example, `run_shell_command(args='ls')`."
        )
    if file_tools:
        tools.append(FileTools(base_dir=cwd))
        extra_instructions.append(
            "You can use the `read_file` tool to read a file, `save_file` to save a file, and `list_files` to list files in the working directory."
        )

    # Add team members available to the LLM OS
    team: List[Assistant] = []
    if data_analyst:
        _data_analyst = DuckDbAssistant(
            name="Data Analyst",
            role="Analyze grades data and provide insights",
            semantic_model=json.dumps(
                {
                    "tables": [
                        {
                            "name": "grades",
                            "description": "CSV of my the grades.",
                            "path": "https://pesugrades.s3.eu-north-1.amazonaws.com/pesu-eval-ipcv.grades.csv",
                        }
                    ]
                }
            ),
            base_dir=scratch_dir,
        )
        team.append(_data_analyst)
        extra_instructions.append(
            "To answer questions about my the details of the grades tabel, delegate the task to the `Data Analyst`."
        )
    if python_assistant:
        _python_assistant = PythonAssistant(
            name="Python Assistant",
            role="Write and run python code",
            pip_install=True,
            charting_libraries=["streamlit"],
            base_dir=scratch_dir,
        )
        team.append(_python_assistant)
        extra_instructions.append("To write and run python code, delegate the task to the `Python Assistant`.")
    if research_assistant:
        _research_assistant = Assistant(
            name="Research Assistant",
            role="Write a research report on a given topic",
            llm=OpenAIChat(model=llm_id),
            description="You are a Senior New York Times researcher tasked with writing a cover story research report.",
            instructions=[
                "For a given topic, use the `search_exa` to get the top 10 search results.",
                "Carefully read the results and generate a final - NYT cover story worthy report in the <report_format> provided below.",
                "Make your report engaging, informative, and well-structured.",
                "Remember: you are writing for the New York Times, so the quality of the report is important.",
            ],
            expected_output=dedent(
                """\
            An engaging, informative, and well-structured report in the following format:
            <report_format>
            ## Title

            - **Overview** Brief introduction of the topic.
            - **Importance** Why is this topic significant now?

            ### Section 1
            - **Detail 1**
            - **Detail 2**

            ### Section 2
            - **Detail 1**
            - **Detail 2**

            ## Conclusion
            - **Summary of report:** Recap of the key findings from the report.
            - **Implications:** What these findings mean for the future.

            ## References
            - [Reference 1](Link to Source)
            - [Reference 2](Link to Source)
            </report_format>
            """
            ),
            tools=[ExaTools(num_results=5, text_length_limit=1000)],
            # This setting tells the LLM to format messages in markdown
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_research_assistant)
        extra_instructions.append(
            "To write a research report, delegate the task to the `Research Assistant`. "
            "Return the report in the <report_format> to the user as is, without any additional text like 'here is the report'."
        )
    if investment_assistant:
        _investment_assistant = Assistant(
            name="Investment Assistant",
            role="Write a investment report on a given company (stock) symbol",
            llm=OpenAIChat(model=llm_id),
            description="You are a Senior Investment Analyst for Goldman Sachs tasked with writing an investment report for a very important client.",
            instructions=[
                "For a given stock symbol, get the stock price, company information, analyst recommendations, and company news",
                "Carefully read the research and generate a final - Goldman Sachs worthy investment report in the <report_format> provided below.",
                "Provide thoughtful insights and recommendations based on the research.",
                "When you share numbers, make sure to include the units (e.g., millions/billions) and currency.",
                "REMEMBER: This report is for a very important client, so the quality of the report is important.",
            ],
            expected_output=dedent(
                """\
            <report_format>
            ## [Company Name]: Investment Report

            ### **Overview**
            {give a brief introduction of the company and why the user should read this report}
            {make this section engaging and create a hook for the reader}

            ### Core Metrics
            {provide a summary of core metrics and show the latest data}
            - Current price: {current price}
            - 52-week high: {52-week high}
            - 52-week low: {52-week low}
            - Market Cap: {Market Cap} in billions
            - P/E Ratio: {P/E Ratio}
            - Earnings per Share: {EPS}
            - 50-day average: {50-day average}
            - 200-day average: {200-day average}
            - Analyst Recommendations: {buy, hold, sell} (number of analysts)

            ### Financial Performance
            {analyze the company's financial performance}

            ### Growth Prospects
            {analyze the company's growth prospects and future potential}

            ### News and Updates
            {summarize relevant news that can impact the stock price}

            ### [Summary]
            {give a summary of the report and what are the key takeaways}

            ### [Recommendation]
            {provide a recommendation on the stock along with a thorough reasoning}

            </report_format>
            """
            ),
            # tools=[YFinanceTools(stock_price=True, company_info=True, analyst_recommendations=True, company_news=True)],
            # This setting tells the LLM to format messages in markdown
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_investment_assistant)
        extra_instructions.extend(
            [
                "To get an investment report on a stock, delegate the task to the `Investment Assistant`. "
                "Return the report in the <report_format> to the user without any additional text like 'here is the report'.",
                "Answer any questions they may have using the information in the report.",
                "Never provide investment advise without the investment report.",
            ]
        )
    
    if project_assistant:
        _project_assistant = Assistant(
            name="Project Assistant",
            role="Provide comprehensive insights and updates on a specific project",
            llm=OpenAIChat(model=llm_id),
            description="As a Senior Project Manager, you are tasked with providing detailed, actionable insights on the progress and performance of a critical project to key stakeholders. The Project Assistant is trained on project-specific documents including the Business Requirement Document, performance reviews, and milestone reports.",
            instructions=[
                "Retrieve and analyze project-specific data from trained documents such as BRD, performance data, and milestone reports.",
                "Generate a detailed report based on the <report_format> below, incorporating data-driven insights and updates.",
                "Provide thoughtful, strategic recommendations for future project steps based on the analyzed data.",
                "Ensure responses and reports include quantitative and qualitative metrics to support project tracking and decision-making.",
                "Maintain high-quality, clear communication to assist stakeholders in understanding the project's progress and strategic value."
            ],
            expected_output=dedent(
                """\
            <report_format>
            ## [Project Name]: Comprehensive Project Update Report

            ### **Overview**
            {Provide a concise introduction of the project, emphasizing its strategic importance and current status based on the BRD and other project documents.}

            ### Milestones Achieved
            {Detail the project's achieved milestones, current phase, and deliverables completed, drawing from milestone reports and performance data.}
            - Current Phase: {current phase of the project}
            - Key Deliverables Completed: {list key deliverables}
            - Upcoming Milestones: {upcoming milestones}
            - Budget Spent: {amount spent} of {total budget}

            ### Performance Metrics
            {Analyze and report on the project's performance, including completion percentages, adherence to timelines, and resource utilization, supported by performance data.}
            - Completion Percentage: {completion percentage}
            - On-Time Delivery: {on-time delivery metrics}
            - Resource Utilization: {resource utilization metrics}

            ### Challenges and Strategic Adjustments
            {Summarize challenges encountered, adjustments made to the project plan, and strategic decisions, using data and insights from project performance reviews.}

            ### [Summary]
            {Provide a summary of the current state of the project, key insights, and takeaways from the data analyzed.}

            ### [Recommendations]
            {Offer strategic and actionable recommendations for the next steps of the project, supported by analyzed data and projected outcomes.}

            </report_format>
            """
            ),
            tools=[YFinanceTools()],

            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_project_assistant)
        extra_instructions.extend(
            [
                "To obtain a project update or detailed report, delegate the task to the `Project Assistant`.",
                "Ensure that the returned report follows the <report_format> and is delivered without additional commentary like 'here is the report'.",
                "Respond to specific inquiries using detailed information from the project's training data and generated reports.",
                "Provide insights and updates consistently within the framework of the provided report, ensuring clarity and depth for stakeholder decision-making."
            ]
        )

    # Create the LLM OS Assistant
    llm_os = Assistant(
        name="llm_os",
        run_id=run_id,
        user_id=user_id,
        llm=OpenAIChat(model=llm_id),
        description=dedent(
            """\
               You are a personalized AI chatbot designed to interact with the Pro Chancellor of PES University, Professor Jawahar Doreswamy. 
               You are equipped with comprehensive information about the PESU Automated Assessments Project.
                As the most advanced AI system in the world, you have access to a suite of tools and a team of AI Assistants. 
                Your primary objective is to provide the best possible assistance to the user in their endeavors related to the project.\
\
        """
        ),
        instructions=[
            "When Professor Jawahar Doreswamy sends a message, first **think** and determine if:\n"
            " - You can answer by using a tool available to you\n"
            " - You need to search the knowledge base\n"
            " - You need to search the internet\n"
            " - You need to delegate the task to a team member\n"
            " - You need to ask a clarifying question",
            " - The answer is already available in the detailed information specific to the PESU Automated Assessments Project.\n"
            " - You need to search Professor Jawahar Doreswamy's specific knowledge base using the `search_knowledge_base` tool.\n"
            " - You need to gather more information from external sources using the `duckduckgo_search` tool.\n"
            " - The query requires the involvement of another AI Assistant with specialized capabilities.\n"
            " - You need to ask clarifying questions to ensure you fully understand the query.",
            "If Professor Doreswamy inquires about a topic related to the PESU Automated Assessments Project, first ALWAYS search his personalized knowledge base using the `search_knowledge_base` tool.",
            "If you do not find relevant information in the knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If Professor Doreswamy requests a summary of the conversation or you need to reference your chat history, use the `get_chat_history` tool.",
            "If the message from Professor Doreswamy is unclear, ask clarifying questions to ensure accurate and useful responses.",
            "After gathering the necessary information, provide a clear and concise answer to Professor Doreswamy.",
            "Avoid using phrases like 'based on my knowledge' or 'depending on the information'. Instead, present the facts as they are.",
            "You can delegate tasks to an AI Assistant in your team depending on their role and the tools available to them, particularly if the task demands specialized expertise.",
        ],
        extra_instructions=extra_instructions,
        # Add long-term memory to the LLM OS backed by a PostgreSQL database
        storage=PgAssistantStorage(table_name="llm_os_runs", db_url=db_url),
        # Add a knowledge base to the LLM OS
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="llm_os_documents",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
            ),
            # 3 references are added to the prompt when searching the knowledge base
            num_documents=3,
        ),
        # Add selected tools to the LLM OS
        tools=tools,
        # Add selected team members to the LLM OS
        team=team,
        # Show tool calls in the chat
        show_tool_calls=True,
        # This setting gives the LLM a tool to search the knowledge base for information
        search_knowledge=True,
        # This setting gives the LLM a tool to get chat history
        read_chat_history=True,
        # This setting adds chat history to the messages
        add_chat_history_to_messages=True,
        # This setting adds 6 previous messages from chat history to the messages sent to the LLM
        num_history_messages=6,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds the current datetime to the instructions
        add_datetime_to_instructions=True,
        # Add an introductory Assistant message
        iintroduction=dedent(
            """\
            Hello, Professor Jawahar Doreswamy, I'm your dedicated AI assistant for the PESU Automated Assessments Project.
            I am equipped with a comprehensive set of tools and a team of AI Assistants specifically aligned to support your project needs.
            Together, we will explore solutions, provide analytics, and enhance the efficiency of your project.\
            """
        ),
        debug_mode=debug_mode,
    )
    return llm_os
