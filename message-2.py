from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
import os

load_dotenv()


def get_weather(city: str) -> str:
    """获取城市的天气"""
    return f"{city}的天气是晴朗的"


with PostgresSaver.from_conn_string(os.getenv("DB_URI")) as checkpointer:

    # checkpointer.setup()

    agent = create_agent(
        model="deepseek-chat",
        tools=[get_weather],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "1"}}

    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "你好, 你还记得我是谁吗?",
                }
            ]
        },
        stream_mode="messages",
        config=config,
    ):
        print(chunk[0].content, flush=True, end="")
