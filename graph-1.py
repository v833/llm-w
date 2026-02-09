from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langmem.utils import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model="deepseek-chat")


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

tools = [get_weather]
tool_node = ToolNode(tools)

def assistant(state: MessagesState, config: RunnableConfig):
    system_prompt = 'You are a helpful assistant that can check weather.'
    all_messages = [SystemMessage(system_prompt)] + state['messages']
    model = llm.bind_tools(tools)
    return {'messages': [model.invoke(all_messages)]}

def should_continue(state: MessagesState, config: RunnableConfig):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'continue'
    return 'end'

builder = StateGraph(MessagesState)
builder.add_node('assistant',assistant)
builder.add_node('tool', tool_node)

builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant',
    should_continue,
    {
        'continue': 'tool',
        'end': END,
    },
)
builder.add_edge('tool', 'assistant')

graph = builder.compile(name="graph_1")

response = graph.invoke({'messages': [HumanMessage(content='上海天气怎么样？')]})
for message in response['messages']:
    message.pretty_print()