from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

checkpointer = InMemorySaver()


def get_weather(city: str) -> str:
    """获取城市的天气"""
    return f"{city}的天气是晴朗的"


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
                "content": "你好, 上海的天气怎么样?",
            }
        ]
    },
    stream_mode="messages",
    config=config,
):
    print(chunk[0].content, flush=True, end="")

for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "你好, 刚才我问了什么?",
            }
        ]
    },
    stream_mode="messages",
    config=config,
):
    print(chunk[0].content, flush=True, end="")
