from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


def get_weather(city: str) -> str:
    """获取城市的天气"""
    return f"{city}的天气是晴朗的"


agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather],
)

# for event in agent.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "你好, 上海的天气怎么样?",
#             }
#         ],
#     },
#     stream_mode="values",
# ):

#     messages = event["messages"]

#     print(messages[-1].pretty_print())


for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "你好, 上海的天气怎么样?",
            }
        ],
    },
    stream_mode="messages",
):
    print(chunk[0].content, flush=True, end="")
