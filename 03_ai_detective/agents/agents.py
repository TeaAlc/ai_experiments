import runnables as r

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable

from .memory import AgentMemory
from helper import read_text_file

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

    def get_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "**YOUR NAME IS {name}**\n\n{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        prompt = prompt.partial(name=self.name, system_prompt=self.system_prompt)
        return prompt

    def get_custom_chain_agent_with_memory(self):
        if self._custom_agent is None:
            chain = (
                    r.extract_input()
                    | r.add_chat_history(self.memory)
                    | r.add_prompt(self.get_prompt())
                    | r.call_model(self.model_w_tools)
                    # maybe a validation step?
                    | r.update_agent_memory(self.memory)
                    | r.reduce_messages()
            )

            self._custom_agent = chain
        return self._custom_agent

    def __call__(self) -> RunnableSerializable[Any, str]:
        return self.get_custom_chain_agent_with_memory()

    @classmethod
    def create(cls, name:str, sys_prompt_path:str, model:BaseChatModel) -> "Agent":
        sys_prompt = read_text_file(sys_prompt_path)
        return Agent(name, sys_prompt, model)


class AgentManager:
    def __init__(self, agents:list[Agent] = None):
        self.agents = {}
        if agents:
            for agent in agents:
                self.put_agent(agent)

    def get_agent(self, agent_name: str) -> Agent:
        return self.agents.get(agent_name)

    def put_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def get_memory(self, agent_name: str) -> AgentMemory:
        return self.agents[agent_name].memory
