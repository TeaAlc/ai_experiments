import os

from langchain_ollama import ChatOllama

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
