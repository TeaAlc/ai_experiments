# LLM TTS Storyteller

An interactive storyteller that generates an endless fantasy narrative using an LLM and reads it aloud with text‑to‑speech (TTS). It supports multi‑turn context with automatic history trimming, multiple interaction modes, helpful console formatting, and a voice selector for playback.

## Features
- Endless, chaptered fantasy story generation.
- Rich console output (colorized names, optional dimmed “hidden” planning blocks).
- Multi‑turn session with message history trimming to keep context within limits.
- Interactive input:
  - Type multi‑line prompts.
  - Submit with two empty lines or Ctrl+D.
  - Choose fully interactive, “first prompt only,” or automatic “go” mode.
- Built‑in validation and retry for LLM responses (balanced tags and curly quotes).
- TTS playback via Kokoro with selectable voices.

## Requirements
- Python 3.12+
- A working virtual environment (recommended).
- An LLM backend:
  - Default: Ollama running locally (or reachable via URL).
  - Alternative: OpenAI‑compatible API (requires code adjustment).
- Audio output device (for TTS).
- System prompt file: `system_prompt.md` (must exist).

## Quick Start

1) Create and activate a virtual environment
- macOS/Linux:
  ```
  python -m venv .venv
  source .venv/bin/activate
  ```
- Windows (PowerShell):
  ```
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

2) Install dependencies
