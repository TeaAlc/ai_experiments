"""Helper utilities for console I/O, LangChain runnables, message validation, and input handling
used by the storyteller application.
"""

import os
import re
import logging
import warnings

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda
from rich.console import Console

CONSOLE = Console(force_terminal=True, color_system="truecolor")
CONSOLE_STDERR = Console(stderr=True, force_terminal=True, color_system="truecolor")


def silence_pytorch_warnings():
    """Reduce PyTorch logging and silence known benign warnings.

    This function:
    - Sets TORCH_CPP_LOG_LEVEL=ERROR to suppress verbose C++ logs.
    - Lowers the log level for torch and torch.distributed.
    - Filters out common RNN/dropout and weight_norm warnings that are not actionable.
    """
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torch\.nn\.modules\.rnn",
        message=r".*dropout.*num_layers=1.*",
    )

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"torch\.nn\.utils\.weight_norm",
    )


def _print_message(message):
    """Print a single, clipped message line in a dim style.

    The output shows up to 100 characters and replaces newlines with a vertical bar separator.
    """
    CONSOLE.print(f"[grey50]\\[{message[:100].replace('\n', ' | ')}][/grey50]")


def _debug_print(data):
    """Pretty-print debug information about a list of messages or a history-like object.

    Args:
        data: Either a list of LangChain message objects or an object with a .messages iterable.
    Returns:
        The original data, unchanged (to allow piping in chains).
    """
    token_count = count_tokens_approximately(data)
    CONSOLE.print("[grey30]--- DEBUG ---[/grey30]")
    CONSOLE.print(f"[grey30]{token_count} tokens in message[/grey30]")
    if isinstance(data, list):
        for message in data:
            message = f"({count_tokens_approximately([message])}) {message.content}"
            _print_message(message)
    else:
        for message in data.messages:
            message = f"({count_tokens_approximately([message])}) {message.content}"
            _print_message(message)

    CONSOLE.print("[grey30]-------------[/grey30]")
    return data


class RunnableLambdas:
    """Reusable RunnableLambda utilities for composing LangChain pipelines.

    Attributes:
        merge: Concatenates chat_history and messages (expects keys in the input mapping).
        debug_print: Prints token counts and message previews, then passes data through.
    """

    merge = RunnableLambda(
        lambda data:
            data["chat_history"] + data["messages"]  # chat_history will be supplied from RunnableWithMessageHistory
    )

    debug_print = RunnableLambda(_debug_print)


def select_interactive_mode() -> int:
    """Interactively prompt the user to select how inputs should be provided.

    Returns:
        1 for fully interactive (type each prompt),
        2 for interactive only for the first prompt,
        3 for automatic (no typing, fixed 'go').
    """
    CONSOLE.print("[bold]Select interaction mode:[/bold]")
    options = [
        ("Interactive (type each prompt)", 1),
        ("Interactive (type first prompt only)", 2),
        ("Automatic (no typing, fixed 'go')", 3),
    ]
    for idx, (label, _) in enumerate(options, start=1):
        CONSOLE.print(f"[cyan]{idx}. {label}[/cyan]")
    while True:
        choice = CONSOLE.input("[green]Mode #> [/green]").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][1]
        CONSOLE.print(f"[red]Please enter a number between 1 and {len(options)}[/red]")


def process_result(message: str, remove_hidden: bool = False) -> str:
    """Convert custom markup in an LLM response to Rich-styled console markup.

    It maps:
    - <color>...</color> to a color span,
    - curly quotes to colored variants,
    - <HIDDEN>/<hidden>/<Hidden> blocks to grey spans (optionally removed).

    Args:
        message: The raw message text to post-process.
        remove_hidden: If True, removes hidden blocks entirely; otherwise renders them dimmed.
    Returns:
        The message with Rich markup suitable for console display.
    """
    if remove_hidden:
        message = re.sub(r"<(Hidden)>.*?</(Hidden)>", "", message, flags=re.DOTALL | re.IGNORECASE)

    message = message.replace("<color>", "[light_slate_blue]").replace("</color>", "[/light_slate_blue]")
    message = message.replace("“", "[dark_khaki]“").replace("”", "”[/dark_khaki]")

    # we tolerate slight errors like hidden instead of HIDDEN
    message = message.replace("<HIDDEN>", "[grey30]").replace("</HIDDEN>", "[/grey30]")
    message = message.replace("<hidden>", "[grey30]").replace("</hidden>", "[/grey30]")
    message = message.replace("<Hidden>", "[grey30]").replace("</Hidden>", "[/grey30]")
    return message


def prepare_speaker_text(message: str) -> str:
    """Strip markup and characters that may degrade TTS pronunciation.

    Removes headings, dashes, color tags, and hidden blocks.

    Args:
        message: The raw message text to sanitize for TTS.
    Returns:
        A plain text string optimized for synthesis.
    """
    message = re.sub(r"<(Hidden)>.*?</Hidden>", "", message, flags=re.DOTALL | re.IGNORECASE)

    message = message.replace("#", "").replace("-", " ").replace("–", "")
    message = message.replace("<color>", "").replace("</color>", "")
    return message


def validate_result(message: str) -> bool:
    """Validate an LLM response against formatting rules required by the app.

    Rules:
    1) No straight double quotes (") are allowed.
    2) <color> tags must be balanced.
    3) <HIDDEN>/<hidden>/<Hidden> tags must be balanced.
    4) Curly quotes (“ and ”) must be balanced.

    Args:
        message: The message to validate.
    Returns:
        True if the message passes all checks; False otherwise (with reason printed to console).
    """
    # 1) Straight double quotes are forbidden
    if "\"" in message:
        cnt = message.count("\"")
        CONSOLE.print(f"[red]Validation failed:[/red] Found {cnt} straight double quote(s) (\") which are forbidden.")
        return False

    # 2) The special name markers must balance
    open_color = message.count("<color>")
    close_color = message.count("</color>")
    if open_color != close_color:
        CONSOLE.print(f"[red]Validation failed:[/red] Unbalanced <color> tags: open={open_color}, close={close_color}.")
        return False

    # 3) Hidden planning block tags must balance (support both lower and upper case)
    open_hidden = message.count("<hidden>") + message.count("<HIDDEN>") + message.count("<Hidden>")
    close_hidden = message.count("</hidden>") + message.count("</HIDDEN>") + message.count("</Hidden>")
    if open_hidden != close_hidden:
        CONSOLE.print(f"[red]Validation failed:[/red] Unbalanced HIDDEN tags: open={open_hidden}, close={close_hidden}.")
        return False

    # 4) Curly quotes must balance
    left_curly = message.count("“")
    right_curly = message.count("”")
    if left_curly != right_curly:
        CONSOLE.print(f"[red]Validation failed:[/red] Unbalanced curly quotes: “={left_curly}, ”={right_curly}.")
        return False

    return True


def drop_last_message(history_store, session_id="1"):
    """Drop the most recent message from a session's history and log a preview.

    Args:
        history_store: A callable that returns a session history object when called with session_id=...
        session_id: The session identifier to operate on.
    """
    session_history = history_store(session_id=session_id)
    message = session_history.messages.pop()
    #CONSOLE.print(f"[red]Dropping last message: {message.content[0:100]}[/red]")


def read_multiline_input(prompt: str = "[green]>>> [/green]") -> str:
    """Read multi-line input from the console with user-friendly submission rules.

    Behavior:
    - Regular Enter inserts a newline and continues reading.
    - Two consecutive empty lines (i.e., pressing Enter twice on an empty line) submit the input.
    - Ctrl+D (EOF) submits whatever has been typed so far.
    - Ctrl+C returns "exit" to signal termination.

    Args:
        prompt: The prompt string to display for the first line.
    Returns:
        The collected multi-line string, or "exit" if the user cancels with EOF/interrupt.
    """
    lines = []
    try:
        first = CONSOLE.input(prompt)
    except (EOFError, KeyboardInterrupt):
        return "exit"

    lines.append(first)
    empty_count = 0
    while True:
        try:
            line = CONSOLE.input()
        except EOFError:
            # Ctrl+D — submit current buffer
            break

        # Count consecutive empty lines; submit after two if we have prior content
        if line == "":
            empty_count += 1
            if empty_count >= 2 and len(lines) > 0:
                break
            # keep waiting for more lines
            continue
        else:
            empty_count = 0

        lines.append(line)

    return "\n".join(lines)
