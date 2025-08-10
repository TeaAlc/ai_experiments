import re
from typing import List, Tuple


def _estimate_text_tokens(text: str) -> int:
    """
    A lightweight token estimator:
    - Splits by words and punctuation to approximate token count.
    - This avoids external dependencies while giving a stable over-approximation.
    """
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
    return len(tokens)


def _estimate_message_tokens(role: str, content: str) -> int:
    """
    Roughly estimate tokens for a single message.
    Adds a small overhead per message to account for formatting/roles.
    """
    overhead = 4  # small constant overhead for role/formatting
    return overhead + _estimate_text_tokens(role) + _estimate_text_tokens(content)


def trim_chat_history(chat_history: List[Tuple[str, str]], max_tokens: int) -> List[Tuple[str, str]]:
    """
    Trim chat history to fit within max_tokens using a simple token estimator.
    - Preserves the most recent messages by working backwards from the end.
    - Returns messages in chronological order.
    - If even the most recent message doesn't fit, returns an empty list.

    Parameters:
        chat_history: list of (role, content) tuples in chronological order.
        max_tokens: maximum allowed token budget for the entire history.

    Returns:
        A new list of (role, content) tuples that fits within the budget.
    """
    if max_tokens <= 0 or not chat_history:
        return []

    # If the most recent message cannot fit, return an empty history
    last_role, last_content = chat_history[-1]
    if _estimate_message_tokens(last_role, last_content) > max_tokens:
        return []

    kept_reversed: List[Tuple[str, str]] = []
    total = 0

    # Greedily keep from newest to oldest; skip any that don't fit
    for role, content in reversed(chat_history):
        msg_tokens = _estimate_message_tokens(role, content)
        if total + msg_tokens <= max_tokens:
            kept_reversed.append((role, content))
            total += msg_tokens

    return list(reversed(kept_reversed))
