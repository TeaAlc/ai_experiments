import os
import sys

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables.base import RunnableBindingBase
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from helpers import CONSOLE, CONSOLE_STDERR, RunnableLambdas, silence_pytorch_warnings, select_interactive_mode, \
    validate_result, drop_last_message, read_multiline_input
from helpers import prepare_speaker_text, process_result
from history_store import ChatHistoryStore

# kokoro produces pytorch warnings, we don't want to see them in our chat!
silence_pytorch_warnings()

from speaker import Speaker, select_speaker_voice


REMOVE_HIDDEN_MESSAGES = True
SHOW_DEBUG_MESSAGES = False

MODEL = "gpt-oss:20b"

# We cut the max token count of the chat history to keep it smooth (~150 Book pages)
CHAT_HISTORY_MAX_TOKEN_COUNT = 8000

# if you want to use Ollama
base_url = os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434")
llm = ChatOllama(model=MODEL, base_url=base_url,
                 num_ctx=round(CHAT_HISTORY_MAX_TOKEN_COUNT * 1.33), num_gpu=999,
                 reasoning="high",
                 )

# If you want to use OpenAI compatible apis
# Hint: Set OPENAI_API_KEY=...
# If using the straico proxy set anything -> OPENAI_API_KEY=whateveryouwant
#llm = ChatOpenAI(model="gpt-4o-mini")
#llm = ChatOpenAI(base_url="http://192.168.178.51:11434", model="openai/gpt-5-nano") # straico over straico proxy (e.G. 192.168.178.51:11434)


# The AI created this ;)
def read_system_prompt(path="system_prompt.md"):
    if not os.path.exists(path):
        CONSOLE_STDERR.print(f"[red]Error: system prompt file '{path}' not found.[/red]")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        CONSOLE_STDERR.print(f"[red]Error: system prompt file '{path}' is empty.[/red]")
        sys.exit(1)
    return content

system_prompt_text = read_system_prompt()
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(system_prompt_text),
    MessagesPlaceholder(variable_name="messages")
])


# we want to trim the max tokens to CHAT_HISTORY_MAX_TOKEN_COUNT, so we throw
# away the oldest messages (exception is the system prompt, we keep it)
trimmer = trim_messages(
    max_tokens=CHAT_HISTORY_MAX_TOKEN_COUNT,
    token_counter=count_tokens_approximately,
    start_on="human",
    end_on=("human", "tool"),
    include_system = True
)

#history_store = ChatHistoryStore()
#history_store(session_id="1").messages.append(HumanMessage("1"))
#history_store(session_id="1").messages.append(HumanMessage("2"))
#history_store(session_id="1").messages.append(HumanMessage("3"))

#chain = merge | trimmer | debug_print | prompt | debug_print
#result = chain.invoke({"messages": [HumanMessage("Hi there")], "chat_history": history_store(session_id="1").messages})

chain = RunnableLambdas.merge | trimmer | prompt | llm
if SHOW_DEBUG_MESSAGES:
    chain = RunnableLambdas.merge | trimmer | prompt | RunnableLambdas.debug_print | llm

# Print the chain graph and see how it looks
# chain.get_graph().print_ascii()

history_store = ChatHistoryStore()
storyteller = RunnableWithMessageHistory(
    chain,
    get_session_history = history_store,
    input_messages_key="messages",
    history_messages_key="chat_history",
)

selected_voice_id = select_speaker_voice()

speaker = Speaker(voice=selected_voice_id)
speaker.start()

interactive_mode = select_interactive_mode()

def retry_invoke(runnable:RunnableBindingBase, question:str, retries:int, session_id="1"):
    exception = None
    while retries > 0:
        retries -= 1
        try:
            result = runnable.invoke(
                {"messages": [HumanMessage(question)]},
                config={"configurable": {"session_id": session_id}},
            )
            if not validate_result(result.content):
                drop_last_message(history_store=history_store, session_id=session_id) # Invalid Answer
                drop_last_message(history_store=history_store, session_id=session_id) # Human Message
                CONSOLE_STDERR.print(f"[red]Invalid result -> retry[/red]")
                exception = BaseException("Invalid result")
                continue
            return result
        except Exception as e:
            drop_last_message(history_store=history_store, session_id=session_id) # Human Message
            CONSOLE_STDERR.print(f"[red]Error on invoke -> retry[/red]")
            exception = e
    raise exception


CONSOLE.print(f"[green]Press Ctrl+D or triple Enter to send message[/green]")

ROUNDS = 15
current_round = 0
exit_words=["exit", "quit", "bye"]
while True:
    current_round += 1

    question = "go"
    if interactive_mode == 1 or (interactive_mode == 2 and current_round == 1):
        question = read_multiline_input()

    if question in exit_words or (current_round == ROUNDS and not interactive_mode == 1):
        break

    result = retry_invoke(storyteller, question,5)
    answer = result.content

    speaker_text = prepare_speaker_text(answer)
    speaker.add_text(speaker_text)

    print_text = process_result(answer, remove_hidden=REMOVE_HIDDEN_MESSAGES)
    CONSOLE.print(print_text, style="blue")

CONSOLE.input("[green]>>> Hit Return To Quit[/green]")
speaker.stop_wait()
