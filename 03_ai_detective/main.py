import os

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.agents import Agent, AgentManager

MODEL = "gpt-oss:20b"

# We cut the max token count of the chat history to keep it smooth (~150 Book pages)
CHAT_HISTORY_MAX_TOKEN_COUNT = 8000

# if you want to use Ollama
base_url = os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434")
llm = ChatOllama(model=MODEL, base_url=base_url,
                 num_ctx=round(CHAT_HISTORY_MAX_TOKEN_COUNT * 1.33), num_gpu=999,
                 reasoning="high",
                 )

# If you want to use OpenAI compatible apis
# Hint: Set OPENAI_API_KEY=...
# If using the straico proxy set anything -> OPENAI_API_KEY=whateveryouwant
#llm = ChatOpenAI(model="gpt-4o-mini")
#llm = ChatOpenAI(base_url="http://192.168.178.51:11434", model="openai/gpt-5-nano") # straico over straico proxy (e.G. 192.168.178.51:11434)


from langchain_core.tools import tool

graph = StateGraph(MessagesState)

agent_tom = Agent(name="Tom", system_prompt="You are Tom, a cat.", model=llm)
agent_jerry = Agent(name="Jerry", system_prompt="You are Jerry, a mouse.", model=llm)
agent_bob = Agent(name="Bob", system_prompt="You are Bob, a dog.", model=llm)

a_manger = AgentManager([agent_tom, agent_jerry, agent_bob])

graph.add_node(agent_tom.name, agent_tom())
graph.add_node(agent_jerry.name, agent_jerry())
graph.add_node(agent_bob.name, agent_bob())


@tool
def speak_with(your_name:str, target_name:str, question:str) -> str:
    """
    Facilitate a conversation turn between two agents.

    Args:
        your_name: The name of the speaking agent initiating the interaction.
        target_name: The name of the target agent to speak with.
        question: The question to ask the target agent.

    Returns:
        A answer to the question or AGENT_NOT_FOUND if the target agent is not found.
    """
    return  "The Answer is 42!"

talk_node = ToolNode([speak_with])

graph.add_node("talk_node", talk_node)

graph.add_edge(agent_tom.name, "talk_node")
graph.add_edge(agent_tom.name, END)

graph.set_entry_point(agent_tom.name)

graph = graph.compile()

#print(graph.get_graph().draw_ascii())

result = graph.invoke({
    "messages": [*agent_tom.memory.get_messages(), HumanMessage("Was ist 12 plus 30?")],
})
print(result["messages"][-1])
