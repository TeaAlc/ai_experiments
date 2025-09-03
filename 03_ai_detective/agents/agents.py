from typing import Any, Optional

from langchain.agents import initialize_agent, AgentType
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, get_buffer_string, SystemMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.utils import Input, Output
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from helper import read_text_file


class AgentMemory(Runnable[dict[str, Any], dict[str, Any]]):
    def __init__(self):
        self._history = InMemoryChatMessageHistory()

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        merged: list[BaseMessage] = self._history.messages + input.get("messages", [])
        return {"messages": merged}

    def get_joined(self) -> str:
        return get_buffer_string(self._history.messages)

    def save(
        self,
        input_msg: Optional[BaseMessage] = None,
        output_msg: Optional[BaseMessage] = None
    ) -> None:
        if input_msg:
            self._history.messages.append(input_msg)
        if output_msg:
            self._history.messages.append(output_msg)

    def get_last_message(self) -> Optional[BaseMessage]:
        return self._history.messages[-1] if self._history.messages else None

    def remove_last_message(self) -> Optional[BaseMessage]:
        if self._history.messages and len(self._history.messages) > 0:
            return self._history.messages.pop()
        return None

    def clear(self) -> None:
        self._history.clear()

    def get_history(self) -> InMemoryChatMessageHistory:
        return self._history


class Agent:
    def __init__(self, name:str, system_prompt:str, model:BaseChatModel, tools=None):
        self.name = name
        self.system_prompt = system_prompt
        self.memory = AgentMemory()
        self.tools = tools or []

        self.model = model
        self.model_w_tools = self.model.bind_tools(self.tools)

        self._react_agent = None
        self._langchain_agent = None
        self._custom_agent = None

    def _build_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "**YOUR NAME IS {name}**\n\n{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])
        prompt.partial(name=self.name, system_prompt=self.system_prompt)
        return prompt

    def _inject_memory(self) -> RunnableLambda:
        def _fn(inputs: dict):
            return {
                **inputs,
                "chat_history": self.memory.messages  # Always pull from memory
            }

        return RunnableLambda(_fn)

    def get_react_agent_without_memory(self):
        if self._react_agent is None:
            agent = create_react_agent(
                model=self.model,
                tools=self.tools,
                prompt= self._build_prompt(),
                name=self.name)
            self._react_agent = self._inject_memory() | agent
        return self._react_agent

    def get_custom_chain_agent_with_memory(self):
        if self._custom_agent is None:
            chain = self._inject_memory() | self._build_prompt() | self.model_w_tools | StrOutputParser()
            self._custom_agent = chain
        return self._custom_agent

    @classmethod
    def create(cls, name:str, sys_prompt_path:str, model:BaseChatModel) -> "Agent":
        sys_prompt = read_text_file(sys_prompt_path)
        return Agent(name, sys_prompt, model)


class AgentManager:
    def __init__(self, agents: dict[str, Agent] = None):
        self.agents = agents or {}

    def get_agent(self, agent_name: str) -> Agent:
        return self.agents.get(agent_name)

    def put_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def get_memory(self, agent_name: str) -> AgentMemory:
        return self.agents[agent_name].memory
