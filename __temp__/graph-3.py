from dataclasses import dataclass
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

basic_llm = init_chat_model(model="deepseek-chat")

@dataclass
class Joke:
    """A joke."""
    setup: str = "A joke setup."
    punchline: str = "A joke punchline."

agent = basic_llm.with_structured_output(Joke)

r = agent.invoke("请讲一个笑话")

print(r)
