
class Supervistor:
    def __init__(self, name:str, system_prompt:str, model:BaseChatModel):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model