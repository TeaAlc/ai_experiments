import unittest

from tokenhelper import (
    trim_chat_history,
    _estimate_text_tokens,
    _estimate_message_tokens,
)


class TestTokenHelper(unittest.TestCase):
    def test_estimate_text_tokens_basic(self):
        self.assertEqual(_estimate_text_tokens(""), 0)
        self.assertEqual(_estimate_text_tokens("hello"), 1)
        self.assertEqual(_estimate_text_tokens("Hello, world!"), 4)  # "Hello", ",", "world", "!"

    def test_estimate_text_tokens_unicode_and_emoji(self):
        # Unicode letters are counted as word tokens
        self.assertEqual(_estimate_text_tokens("naïve café"), 2)
        # Em dash and pi should tokenize as punctuation and word respectively
        self.assertEqual(_estimate_text_tokens("naïve — π"), 3)
        # Emoji is a non-word token, counted separately
        self.assertEqual(_estimate_text_tokens("hi 👋"), 2)

    def test_estimate_message_tokens_overhead(self):
        # Overhead + role tokens + content tokens
        expected = 4 + _estimate_text_tokens("human") + _estimate_text_tokens("Hello, world!")
        self.assertEqual(_estimate_message_tokens("human", "Hello, world!"), expected)

    def test_trim_chat_history_within_budget_keeps_all(self):
        chat = [
            ("human", "one"),
            ("system", "two three"),
            ("human", "Hello, world!"),
        ]
        total = sum(_estimate_message_tokens(r, c) for r, c in chat)
        trimmed = trim_chat_history(chat, total)
        self.assertEqual(trimmed, chat)

    def test_trim_chat_history_upper_border_exact(self):
        """
        Explicitly test the upper border: when max_tokens equals the exact sum
        of the last N messages, those N messages must be kept and no more.
        """
        chat = [
            ("human", "one"),
            ("system", "two three"),
            ("human", "Hello, world!"),
        ]
        last_two = chat[1:]
        border = sum(_estimate_message_tokens(r, c) for r, c in last_two)
        trimmed = trim_chat_history(chat, border)
        self.assertEqual(trimmed, last_two)

    def test_trim_chat_history_returns_empty_if_last_too_large(self):
        chat = [
            ("human", "one"),
            ("system", "two three"),
            ("human", "Hello, world!"),
        ]
        last_tokens = _estimate_message_tokens(*chat[-1])
        trimmed = trim_chat_history(chat, last_tokens - 1)  # budget just below the last message size
        self.assertEqual(trimmed, [])

    def test_trim_chat_history_zero_or_negative_budget(self):
        chat = [("human", "hi")]
        self.assertEqual(trim_chat_history(chat, 0), [])
        self.assertEqual(trim_chat_history(chat, -5), [])

    def test_trim_chat_history_preserves_order_and_prefers_recent(self):
        chat = [
            ("human", "A"),
            ("system", "B"),
            ("human", "C"),
        ]
        # Fit only the last two messages
        budget = sum(_estimate_message_tokens(r, c) for r, c in chat[1:])
        trimmed = trim_chat_history(chat, budget)
        self.assertEqual(trimmed, [("system", "B"), ("human", "C")])

    def test_trim_chat_history_removes_oldest_first_fifo(self):
        """
        When the budget fits only the last N messages, the oldest messages should be removed first (FIFO),
        ensuring the kept messages are the most recent contiguous suffix (not LIFO).
        """
        chat = [
            ("human", "A"),
            ("system", "BB"),
            ("human", "CCC"),
            ("system", "DDDD"),
        ]
        # Budget fits only the last three messages -> the oldest one must be dropped
        budget = sum(_estimate_message_tokens(r, c) for r, c in chat[1:])
        trimmed = trim_chat_history(chat, budget)
        self.assertEqual(trimmed, chat[1:])


if __name__ == "__main__":
    unittest.main()
