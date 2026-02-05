from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    temperature=0.7,
    base_url="https://openai.deepseek.cn/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    timeout=30,
    max_tokens=2000,
)


for chunk in llm.stream("你好, 你是什么模型?"):
    print(chunk.content, flush=True, end="")
