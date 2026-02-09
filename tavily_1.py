from typing import Literal
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

tavily_client = TavilyClient()


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """
    搜索互联网
    """
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


research_instructions = """
你是一个专业的研究助手，能够根据用户的问题搜索互联网并提供相关信息。
你可以使用internet_search函数来搜索互联网。
"""


agent = create_agent(
    model="deepseek-chat",
    tools=[internet_search],
    system_prompt=research_instructions,
)

for step in agent.stream(
    {"messages": [{"role": "user", "content": "什么是langgraph?请详细介绍"}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
