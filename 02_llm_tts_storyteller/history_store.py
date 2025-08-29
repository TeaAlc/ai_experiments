"""Thread-safe chat history store for LangChain sessions.

Exposes a callable interface to retrieve or create per-session histories backed by
InMemoryChatMessageHistory. Designed to be used with RunnableWithMessageHistory.
"""

from typing import Dict
from threading import Lock
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory


class ChatHistoryStore:
    """A thread-safe registry for per-session chat histories.

    Each unique session_id maps to its own BaseChatMessageHistory instance.
    The store can be called like a function to retrieve the history for a session.
    """

    def __init__(self):
        """Initialize the in-memory store and its synchronization primitive."""
        self._store: Dict[str, BaseChatMessageHistory] = {}
        self._lock = Lock()

    def __call__(self, session_id: str) -> BaseChatMessageHistory:
        """Return the chat history for the given session, creating it if necessary.

        This method is thread-safe and ensures that at most one history is created
        per session_id even under concurrent access.

        Args:
            session_id: The identifier of the conversation session.
        Returns:
            A BaseChatMessageHistory instance for the session.
        """
        # Thread-safe "get or create"
        with self._lock:
            hist = self._store.get(session_id)
            if hist is None:
                hist = InMemoryChatMessageHistory()
                self._store[session_id] = hist
            return hist
