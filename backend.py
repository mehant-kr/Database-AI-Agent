from langchain_community.utilities import SQLDatabase
import psycopg2
import getpass
import os
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain import hub
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import TypedDict, Annotated
# create a secret.py file with these variables
from secrets import host, databaseName, username, port, pwd, openAI_api_key

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

os.environ["OPENAI_API_KEY"] = openAI_api_key



def postgresql_db():

    engine = create_engine(f"postgresql+psycopg2://{username}:{pwd}@{host}:{port}/{databaseName}")

    # Get the dialect
    dialect = engine.dialect.name
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    print("Dialect:", dialect)
    print("Table Info:", table_names)
    db = SQLDatabase(engine)
    return db, engine, dialect, table_names


def sqlite_db():
    # db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    dialect = db.dialect
    table_info = db.table_info
    print(dialect)
    print(table_info)

    # List all tables in the database
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = db.run(tables_query)
    print("List of tables in the database:")
    print(tables)

    db.run("SELECT * FROM Artist LIMIT 10;")
    return db, engine, dialect, table_info

db, engine, dialect, table_names = postgresql_db()

# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


llm = ChatOpenAI(model="gpt-4o-mini")



query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

def convert_to_postgresql(sql_statement):
    """if a keyword starts with camel case then put it in inverted commas"""
    import re

    # Sample SQL statement
    # sql_statement = "SELECT ColumnName, AnotherColumn FROM TableName WHERE SomeValue = 10;"

    # Regular expression pattern to match PascalCase words
    pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b'

    # Function to add double quotes around the match
    def add_quotes(match):
        # print("match: " , match)
        return f'"{match.group(0)}"'

    # Replace PascalCase words with quoted versions
    modified_sql = re.sub(pattern, add_quotes, sql_statement)
    return modified_sql

def write_query(state):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": dialect,
            "top_k": 10,
            "table_info": table_names,
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    converted_query = convert_to_postgresql(result["query"])
    return {"query": converted_query}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=SQLDatabase(engine))
    return

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}

# Initialize the message history
message_history = []

def agents(query):
    """
    Create a retriever tool
    """

    # remembering history
    thread = {"Configurable"}

    # Initialize the SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    return_list = []
    # Pull the system prompt template and format it for PostgreSQL
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    system_message = prompt_template.format(dialect="postgresql", top_k=5)

    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

    # Add the new user query to the message history
    message_history.append({"role": "user", "content": query})

    for step in agent_executor.stream(
            {"messages": message_history},
            stream_mode="values",
    ):
        return_list.append(step["messages"][-1])
        # step["messages"][-1].pritty_print()

    # Add the agent's response to the message history
    message_history.append({"role": "assistant", "content": return_list[-1].content})

    return return_list[-1].content
    # return return_list


# result = agents("which country do you think can be a good prospect for selling tracks and why?")
# print(result)

# def dummy_agents(query):
#     return f"This is a dummy response: {query}"
