from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task,entrypoint
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model="deepseek-chat")

checkpointer = InMemorySaver()

@task
def gen_joke(topic:str):
    """Generate a joke about a given topic."""
    result = llm.invoke(f"请生成关于{topic}的笑话")
    return result.content


@entrypoint(checkpointer=checkpointer)
def workflow(topics: list[str]):
    """Generate jokes about a list of topics."""
    futures = [gen_joke(topic) for topic in topics]
    paragraphs = [f.result() for f in futures]
    return '\n\n'.join(paragraphs)

config={"configurable": {"thread_id": "1"}}
result=workflow.invoke(["程序员", "会计", "厨师"], config=config)

print(result)