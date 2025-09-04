from typing import Optional, Any

from langchain_core.runnables.utils import Input, Output
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig, Runnable


class AgentMemory(Runnable[dict[str, Any], dict[str, Any]]):
    def __init__(self):
        self._history = InMemoryChatMessageHistory()

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        merged: list[BaseMessage] = self.get_messages() + input.get("messages", [])
        return {"messages": merged}

    def get_joined(self) -> str:
        return get_buffer_string(self.get_messages())

    def save(
        self,
        input_msg: Optional[BaseMessage] = None,
        output_msg: Optional[BaseMessage] = None
    ) -> None:
        if input_msg:
            self.get_messages().append(input_msg)
        if output_msg:
            self.get_messages().append(output_msg)

    def get_messages(self) -> list[BaseMessage]:
        return self._history.messages

    def get_last_message(self) -> Optional[BaseMessage]:
        return self.get_messages()[-1] if self.get_messages() else None

    def remove_last_message(self) -> Optional[BaseMessage]:
        if self.get_messages() and len(self.get_messages()) > 0:
            return self.get_messages().pop()
        return None

    def clear(self) -> None:
        self._history.clear()

    def get_history(self) -> InMemoryChatMessageHistory:
        return self._history
