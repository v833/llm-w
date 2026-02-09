from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langchain.agents.middleware import wrap_model_call
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

basic_llm = init_chat_model(model="deepseek-chat")

advanced_llm = init_chat_model(model="deepseek-reasoner")


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    根据请求内容动态选择模型
    """
    message_count = len(request.state["messages"])
    if message_count > 5:
        model = basic_llm
    else:
        model = advanced_llm
    
    print(f"message_count:{message_count}, model:{model.model_name}")

    return handler(request.override(model=model))

checkpointer = InMemorySaver()

agent = create_agent(
    model=advanced_llm,
    middleware=[dynamic_model_selection],
    checkpointer=checkpointer,
)

config={"configurable": {"thread_id": "1"}}

state: MessagesState = {"messages": []}
items = ['汽车', '飞机', '摩托车', '自行车']
for idx, i in enumerate(items):
    print(f"\n=== Round {idx+1} ===")
    state["messages"] += [HumanMessage(content=f"{i}有几个轮子，请简单回答")]
    result = agent.invoke(state, config)
    print(f'content: {result["messages"][-1].content}')