from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import dynamic_prompt
from langgraph.store.memory import InMemoryStore
load_dotenv()

@dataclass
class Context:
    """上下文"""
    user_id: str

store = InMemoryStore()

store.put('prefs', 'user_1', {"language": 'zh-CN'})
store.put('prefs', 'user_2', {"language": 'en-US'})

@dynamic_prompt
def get_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    prefs = request.runtime.store.get('prefs', user_id)
    language = prefs.value.get('language', 'zh-CN')
    return f"你是一个{language}的助手, 请使用{language}回答"

agent = create_agent(model="deepseek-chat", middleware=[get_prompt], context_schema=Context, store=store)

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
    context=Context(user_id='user_1'),
):
    print(chunk[0].content, flush=True, end="")