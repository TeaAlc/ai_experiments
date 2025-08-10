# python
"""
Simple RAG CLI using Ollama + LangChain.

- Indexes local documents (./documents) in an in-memory vector store using bge-m3 embeddings via Ollama.
- Enriches user questions with chat history and retrieves relevant context rendered via a Jinja2 template.
- Generates answers with a local LLM (gpt-oss:20b via Ollama).
- Configure the Ollama server with OLLAMA_SERVER_URL (default: http://localhost:11434).
"""

import os

from jinja2 import Environment

from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama

# We have created this completely with AI including unittests
from tokenhelper import trim_chat_history

base_url = os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434")

CONSOLE = Console(force_terminal=True, color_system="truecolor")

# We cut the max token count of the chat history to keep it smooth
CHAT_HISTORY_MAX_TOKEN_COUNT = 32000

# Compare the different results with and without the context enrichment
RAG_MODE_ENRICH_WITH_CONTEXT = True
RAG_PRINT_QUESTION = True
RAG_SUMMARY_SYSTEM_PROMPT = """
You are professional summary writer for summaries used do enrich queries to vector storage
for RAG. You write short summaries of max 5 sentences and max 200 words corresponding to
the question of the user and the previous conversation, also add appropriate information not directly
associated with the question. You must not use any information that is not part of the conversation.
You don't return anything else than the summary.
"""

SYSTEM_PROMPT = """
You are Thadeus Un, a historian, born 50 years ago in the domic fires of Underville.
You are a specialist for symbarian history (The history of Symbaroum), ambrien history, albertian history and
the history of the davokar wood. You are a pretty good story teller, you like putting information into small stories or tales.
Also you love to tell stories and tales, you always stick to the facts you know.
You answer in 1-10 sentences or 1-3 paragraphs, unless you are told other, e.g. tell me a story, telle me a tale, tell me a long ..., go into detail ...
You always structure your answers as markdown, under every circumstances you must structure your answer in markdown.
You always play your role, you won't do things that do not fit this role. You also always speak and answer like an old historian.
"""

embeddings = OllamaEmbeddings(model="bge-m3", base_url=base_url)
vectorstore = InMemoryVectorStore(embedding=embeddings)

llm = ChatOllama(model="gpt-oss:20b", base_url=base_url)

# Read all files from the 'documents' directory into a dictionary: filename -> filecontent
documents_dir = Path(__file__).parent / "documents"
documents: dict[str, str] = {}

print("Reading documents and adding them to the vectorstore")
for file_path in documents_dir.iterdir():
    if file_path.is_file():
        documents[file_path.name] = file_path.read_text(encoding="utf-8")

for k, v in documents.items():
    vectorstore.add_texts([v], [{"filename": k}])
    print(f" Added {k} to vectorstore")


def get_rag(question:str, vectorstore:VectorStore, llm:BaseChatModel,  chat_history=[],
            enrich_with_context=True, print_question=True,
            threshold=0.25, k=3) -> str:
    """
    Construct a rendered context string for Retrieval-Augmented Generation (RAG).

    Optionally summarizes recent chat history to enrich the question (when enabled), retrieves
    the top-k most similar documents from the vector store, filters them by a similarity score
    threshold, and renders the selected results using a Jinja2 template.

    Args:
        question (str): The user's question to be contextualized.
        vectorstore (VectorStore): Vector store used to perform similarity search.
        llm (BaseChatModel): Chat model used to summarize/enrich the question when chat history exists.
        chat_history (list[tuple[str, str]]): Prior conversation as (role, content) pairs; used for enrichment.
        enrich_with_context (bool): If True, use the LLM to summarize chat_history and prepend it to the question.
        print_question (bool): If True, print the question and any included documents to the console.
        threshold (float): Minimum similarity score (inclusive) required to include a document in the context.
        k (int): Number of top results to retrieve from the vector store.

    Returns:
        str: Rendered context string intended to precede the assistant's answer.
    """

    # We enrich the question if there was a chat already, to provide more context for RAG
    if len(chat_history) > 1 and enrich_with_context:
        chat = [("system", RAG_SUMMARY_SYSTEM_PROMPT)] + chat_history + [("human",f"Create a summary corresponding to the question: {question}")]
        result = llm.invoke(chat)
        question = f"{result.content}\n\n{question}"

    if print_question:
        CONSOLE.print(f"RAG QUESTION: {question}", style="grey37")
    results = vectorstore.similarity_search_with_score(
        question, k=k
    )

    templatestring = (Path(__file__).parent / "ragtemplate.jinja2").read_text(encoding="utf-8")
    template = Environment().from_string(templatestring)

    # Build a list of plain dicts for the template and throw out documents with a similarity score < threshold
    results_for_template = []
    for doc, score in results:
        if score >= threshold:
            results_for_template.append(
                {
                    "metadata": doc.metadata,
                    "page_content": doc.page_content,
                    "score": score,
                }
            )

    if print_question:
        for result in results_for_template:
            CONSOLE.print(f"- added {result["metadata"]["filename"]} ({result["score"]})", style="grey37")

    ragdata = template.render(results=results_for_template)
    return ragdata

complete_chat_history = []  # Contains the whole chat history without any cutoff e.G. usable for creating a long-term memory
chat_history = []           # Always contains as much chat history as possible in the given token limit
question = None

exit_words=["exit", "quit", "bye"]
while True:
    question = CONSOLE.input("[green]>>> [/green]")

    if question in exit_words:
        quit(0)

    # get the data to argument the request with the provided additional knowledge
    ragdata = get_rag(question, llm=llm, vectorstore=vectorstore, chat_history=chat_history,
                      enrich_with_context=RAG_MODE_ENRICH_WITH_CONTEXT,
                      print_question=RAG_PRINT_QUESTION)

    messages = ([("system", SYSTEM_PROMPT)] + chat_history +
                [ ("assistant", ragdata),
                  ("human", question)])

    result = llm.invoke(messages)
    answer = result.content
    CONSOLE.print(Markdown(answer), style="blue")

    complete_chat_history.append(("human", question))
    complete_chat_history.append(("assistant", answer))
    chat_history.append(("human", question))
    chat_history.append(("assistant", answer))

    chat_history = trim_chat_history(chat_history=chat_history, max_tokens=32000)
