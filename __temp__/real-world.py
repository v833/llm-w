from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

load_dotenv()

SYSTEM_PROMPT = """你是一位专业的天气预报专家，说话时喜欢使用双关语和俏皮话。

你可以使用以下两个工具：

get_weather_for_location：用来获取特定地点的天气信息

get_user_location：用来获取用户所在的位置

如果用户向你询问天气情况，请确保你知道具体地点。如果从问题中可以判断他们指的是自己所在的位置，请使用get_user_location工具来确定他们的位置。"""


@dataclass
class Context:
    """上下文类，用于存储用户ID"""

    user_id: str


@dataclass
class ResponseFormat:
    """Response schema for the agent."""

    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


@tool
def get_weather_for_location(location: str) -> str:
    """获取特定地点的天气信息"""
    return f"{location}的天气是晴朗的"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """判断用户所在的位置"""
    return "上海" if runtime.context.user_id == "1" else "北京"


checkpointer = InMemorySaver()

config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model="deepseek-chat",
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather_for_location, get_user_location],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_user_location": True,
                "get_weather_for_location": {
                    "allowed_decisions": ["approve", "reject"]
                },
            },
            description_prefix="你是否同意执行以下操作？",
        )
    ],
    checkpointer=checkpointer,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "你好, 外面的天气怎么样?"},
        ]
    },
    config=config,
    context=Context(user_id="2"),
)

# print(result["structured_response"])

for message in result["messages"]:
    print(message.pretty_print())

    if "__interrupt__" in result:
        print("__interrupt__")
        interrupt = result["__interrupt__"][0]
        for decision in interrupt.value["action_requests"]:
            print(decision["description"])

result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),  # or "reject"
    config=config,  # Same thread ID to resume the paused conversation
    context=Context(user_id="2"),
)
