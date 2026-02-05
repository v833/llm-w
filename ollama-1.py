from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen3-vl:8b",
    model_provider="ollama",
    temperature=0.7,
    base_url="http://localhost:11434",
)

try:
    for chunk in llm.stream("你好, 你是什么模型?"):
        print(chunk.content, flush=True, end="")
except Exception as e:
    print(f"发生错误: {e}")
