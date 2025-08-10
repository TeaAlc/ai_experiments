# Simple RAG with Ollama + LangChain

A minimal Retrieval-Augmented Generation (RAG) scaffold that:
- Loads local documents into an in-memory vector store
- Uses `bge-m3` embeddings via Ollama
- Renders the RAG results through a Jinja2 template
- Queries a local LLM (`gpt-oss:20b` via Ollama) to enrich the question and to interact

This project is intended as a small, hackable starting point for experimenting with local RAG setups.

## Prerequisites
- Python 3.12+
- GPU with 16 GB VRAM recommended for optimal performance with `gpt-oss:20b`
  - CPU-only or smaller-GPU setups may work but will be slower and potentially memory-constrained.
- A shell/terminal with network access to the Ollama server (default: `http://localhost:11434`)
- Optional but recommended: a virtual environment manager (e.g., `venv`)

## Preparation (Ollama + Models)
### 1) Install Ollama
- Windows: Download the installer from https://ollama.com/download.
- Linux:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- macOS: Download the app from https://ollama.com/download and follow the installer.

After installation, ensure the Ollama service is running:
- Windows: Starts with the app (or run `ollama serve` in a terminal).
- Linux (systemd): `systemctl --user start ollama` (or run `ollama serve` in a terminal).
- macOS: Launches with the app.

If you want to use different servers, one with ollama, and one for coding. (I recommend this)<br>
- Make ollama listening on all interfaces on linux `export OLLAMA_HOST=0.0.0.0:11434`
- For windows use the `Advanced System Settings>Environment Variables` and set `OLLAMA_HOST` to `0.0.0.0:11434`
- Change `base_url` in `main.tf` to your server or set `OLLAMA_SERVER_URL`

### Install the requirements
**Hint:** It is recommended to use a [venv](https://docs.python.org/3/library/venv.html) for this.
```bash
pip install -r requirements.txt
```
