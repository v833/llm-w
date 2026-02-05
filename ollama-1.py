from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="ollama:qwen3-vl:8b",
    temperature=0.7,
    base_url="http://localhost:11434",
    timeout=30,
    max_token=2000,
)

try:
    for chunk in llm.stream("你好, 你是什么模型?"):
        print(chunk.content, flush=True, end="")
except Exception as e:
    print(f"发生错误: {e}")
