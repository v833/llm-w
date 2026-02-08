from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
import os

load_dotenv()


def get_weather(city: str) -> str:
    """获取城市的天气"""
    return f"{city}的天气是晴朗的"


with PostgresSaver.from_conn_string(os.getenv("DB_URI")) as checkpointer:

    config = {"configurable": {"thread_id": "1"}}

    checkpoints = checkpointer.list(config)

    for checkpoint in checkpoints:
        messages = checkpoint[1]["channel_values"]["messages"]
        for message in messages:
            message.pretty_print()
        break
    # agent = create_agent(
    #     model="deepseek-chat",
    #     tools=[get_weather],
    #     checkpointer=checkpointer,
    # )
