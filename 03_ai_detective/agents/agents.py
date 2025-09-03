from langchain_core.language_models import BaseChatModel

from helper import read_text_file


class Agent:
    def __init__(self, name:str, system_prompt:str, llm:BaseChatModel):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm

    @classmethod
    def create(name:str, sys_prompt_path:str, llm:BaseChatModel) -> "Agent":
        sys_prompt = read_text_file(sys_prompt_path)
        return Agent(name, sys_prompt, llm)
