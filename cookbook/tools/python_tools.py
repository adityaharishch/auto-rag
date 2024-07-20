from micro.assistant import Assistant
from micro.tools.python import PythonTools

assistant = Assistant(tools=[PythonTools()], show_tool_calls=True)
assistant.print_response(
    "Write a python script for fibonacci series and display the result till the 10th number", markdown=True
)
