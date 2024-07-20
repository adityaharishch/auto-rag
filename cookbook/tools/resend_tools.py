from micro.assistant import Assistant
from micro.tools.resend_tools import ResendTools

assistant = Assistant(tools=[ResendTools(from_email="<enter_from_email>")], debug_mode=True)

assistant.print_response("send email to <enter_to_email> greeting them with hello world")
