from langchain.agents import initialize_agent, AgentType
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from helper import read_text_file


class Agent:
    def __init__(self, name:str, system_prompt:str, model:BaseChatModel, tools=[]):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools

        self.model_w_tools = self.model.bind_tools(self.tools)

        self._react_agent = None
        self._langchain_agent = None
        self._custom_agent = None

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "system",
                f"**YOUR NAME IS {self.name}**\n\n{self.system_prompt}"
            ),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

    def get_langchain_agent(self):
        if self._langchain_agent is None:
            self._langchain_agent = initialize_agent(
                llm=self.model,
                tools=self.tools,
                agent=AgentType.OPENAI_FUNCTIONS,
                agent_kwargs={"prompt": self._build_prompt()},
                verbose=True
            )
        return self._langchain_agent

    def get_react_agent(self):
        if self._react_agent is None:
            self._react_agent = create_react_agent(
                model=self.model,
                tools=self.tools,
                prompt= self._build_prompt(),
                name=self.name)

    def get_custom_chain_agent(self):
        if self._custom_agent is None:
            chain = self._build_prompt() | self.model_w_tools | StrOutputParser()
            self._custom_agent = chain
        return self._custom_agent

    @classmethod
    def create(cls, name:str, sys_prompt_path:str, model:BaseChatModel) -> "Agent":
        sys_prompt = read_text_file(sys_prompt_path)
        return Agent(name, sys_prompt, model)
