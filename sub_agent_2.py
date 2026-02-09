from langchain.tools import tool
from langchain.agents import create_agent

import dotenv

dotenv.load_dotenv()

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)


@tool
def send_email(
    to: list[str], subject: str, body: str, cc: list[str] = []  # email addresses
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


email_agent = create_agent(
    model="deepseek-chat",
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)
