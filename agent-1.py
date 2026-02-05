from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()

agent = create_agent(
    model="deepseek-chat",
)

results = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "你好",
            }
        ]
    }
)

messages = results["messages"]


for message in messages:
    message.pretty_print()
