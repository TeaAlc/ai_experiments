from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable

from agents.memory import AgentMemory


def extract_input() -> RunnableLambda:
    def _fn(data: dict):
        return {
            **data,
            "input": data["messages"][-1].content  # The last message content is the input
        }

    return RunnableLambda(_fn)

def add_chat_history(memory:AgentMemory) -> RunnableLambda:
    def _fn(data: dict):
        return {
            **data,
            "chat_history": memory.get_messages()  # Always pull from memory
        }

    return RunnableLambda(_fn)

def add_prompt(template:ChatPromptTemplate) -> RunnableLambda:
    def _fn(data: dict):
        prompt = template.invoke(data)
        return {
            **data,
            "prompt": prompt  # The PromptValue
        }

    return RunnableLambda(_fn)

def call_model(model:Runnable) -> RunnableLambda:
    def _fn(data: dict):
        call_result = model.invoke(data["messages"])
        messages = data["messages"]
        messages.append(AIMessage(call_result.content))
        return {
            **data,
            "call_result": call_result,
            "output": data["messages"][-1].content  # The last message content is the input
        }

    return RunnableLambda(_fn)

def update_agent_memory(memory:AgentMemory) -> RunnableLambda:
    def _fn(data: dict):
        memory.save(input_msg=data["input"], output_msg=data["output"])
        return {
            **data,
        }

    return RunnableLambda(_fn)

def reduce_messages() -> RunnableLambda:
    def _fn(data: dict):
        messages = data["messages"]
        messages = messages[-2:]
        data["messages"] = messages
        return {
            **data,
        }

    return RunnableLambda(_fn)
