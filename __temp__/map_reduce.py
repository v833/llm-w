from pydantic import BaseModel
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.constants import START
from langgraph.graph import END
from langgraph.graph.state import StateGraph
from langgraph.types import Send
from typing_extensions import Annotated
import operator
from dotenv import load_dotenv

load_dotenv()


# 角色
class Roles(BaseModel):
    roles: list[str]


# 单个角色
class Role(BaseModel):
    role: str


# 单个回复
class Response(BaseModel):
    response: str


# 最佳回复的 ID
class BestResponse(BaseModel):
    id: int


# 全局上下文
class Overall(TypedDict):
    roles: list[str]
    responses: Annotated[list, operator.add]
    best_response: str


# 角色的提示词
role_prompt = "女神又不回你消息了，作为一个{role}，你应该如何一句话回复女神？请以JSON格式返回，包含response字段"

# 最佳回复的提示词
best_response_prompt = """下面是几种类型的男生，面对女神不回消息的情况，做出的反应。
你觉得以下哪种回复最能挽回女神的心，请返回对应的ID。
注意哦，第一条反应对应的是0号ID。并以JSON格式返回，包含id字段
下面是男生们的反应：\n\n{responses}"""

llm = init_chat_model(model="deepseek-chat")


def continue_to_responses(state: Overall):
    return [Send("generate_response", {"role": r}) for r in state["roles"]]


def generate_response(state: Role):
    prompt = role_prompt.format(role=state["role"])
    response = llm.with_structured_output(Response).invoke(prompt)
    return {"responses": [response.response]}


def best_response(state: Overall):
    responses = "\n\n".join(state["responses"])
    prompt = best_response_prompt.format(responses=responses)
    response = llm.with_structured_output(BestResponse).invoke(prompt)
    return {"best_response": state["responses"][response.id]}


# 定义Doge子图的输出Schema
class DogeOutput(TypedDict):
    roles: list[str]
    responses: list[str]
    best_response: str


doge_builder = StateGraph(Overall, output_schema=DogeOutput)


doge_builder.add_node(generate_response)
doge_builder.add_node(best_response)

doge_builder.add_conditional_edges(START, continue_to_responses, ["generate_response"])
doge_builder.add_edge("generate_response", "best_response")
doge_builder.add_edge("best_response", END)

doge_graph = doge_builder.compile(name="best-response")

roles = [
    "男神",
    "巨魔",
    "舔狗",
    "渣男",
    "奶狗弟弟",
    "社恐宅男",
    "霸道总裁",
    "茶茶的男生 💅💅💅",
    "文艺长发男",
    "萌萌二次元",
]
response = doge_graph.invoke({"roles": roles})
print(f"最受喜爱的回复：{response['best_response']}")
