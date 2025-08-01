## Imports ??
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from langgraph.graph.state import StateGraph
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

## Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "LangGraphMonitoring"


## Initialize the LLM
from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama-3.3-70b-versatile")


## Define the state for the graph
class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def make_tool_graph():
    ## Graph With tool Call
    
    @tool
    def add(a:float,b:float):
        """Add two number"""
        return a+b
    
    tools=[add]
    
    tool_node=ToolNode(tools=tools)

    llm_with_tools = llm.bind_tools(tools=tools)

    def call_llm_model(state:State):
        return {
            "messages":[llm_with_tools.invoke(state['messages'])]
        }
    

    ## Grpah
    builder=StateGraph(State)
    
    ## Add Nodes
    builder.add_node("LLM Call", call_llm_model)
    builder.add_node("tools", tool_node)

    ## Add Edges
    builder.add_edge(START, "LLM Call")
    builder.add_conditional_edges(
        "LLM Call",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools","LLM Call")

    ## compile the graph
    graph = builder.compile()
    
    return graph


## Create the tool agent graph
## Note: This graph is used to monitor the tool calls made by the agent.
tool_agent = make_tool_graph()

