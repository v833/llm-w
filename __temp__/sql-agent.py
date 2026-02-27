from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
import requests, pathlib
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

load_dotenv()


url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"数据库文件 {local_path} 已存在")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"数据库文件 {local_path} 下载完成")
    else:
        print(f"下载数据库文件 {local_path} 失败，状态码: {response.status_code}")

db = SQLDatabase.from_uri(f"sqlite:///{local_path}")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')

llm = init_chat_model(model="deepseek-chat")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

# for tool in tools:
#     print(f"Tool: {tool.name}")
#     print(f"Description: {tool.description}")
#     print("=" * 50)

system_prompt = """
使用中文回答问题
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="等待用户确认SQL查询",
        )
    ],
    checkpointer=checkpointer,
)

question = "Which genre on average has the longest tracks?"
config = {"configurable": {"thread_id": "1"}}


for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
    config=config,
):
    if "__interrupt__" in step:
        print("__interrupt__")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
        user_input = input("\n请输入确认或拒绝(approve/reject): ").strip().lower()
        if user_input == "approve":
            command = Command(resume={"decisions": [{"type": "approve"}]})
        elif user_input == "reject":
            command = Command(resume={"decisions": [{"type": "reject"}]})
        else:
            print("无效输入，请输入 approve 或 reject")
            continue
    elif "messages" in step:
        step["messages"][-1].pretty_print()
    else:
        pass


for step in agent.stream(
    command,
    config=config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    else:
        pass
