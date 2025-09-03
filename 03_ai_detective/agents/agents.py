from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from helper import read_text_file


class Agent:
    def __init__(self, name:str, system_prompt:str, model:BaseChatModel):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

        self._react_agent = None

    def get_react_agent(self):
        if self._react_agent is None:
            self._react_agent = create_react_agent(
                model=self.model,
                tools=[],
                prompt= f"**YOUR NAME IS {self.name}**\n\n{self.system_prompt}",
                name=self.name)

    @classmethod
    def create(cls, name:str, sys_prompt_path:str, model:BaseChatModel) -> "Agent":
        sys_prompt = read_text_file(sys_prompt_path)
        return Agent(name, sys_prompt, model)
