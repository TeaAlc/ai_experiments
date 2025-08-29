"""Text-to-speech speaker utilities using Kokoro TTS and sounddevice playback.

Provides an interactive voice selector and a Speaker class that synthesizes text
to audio in a background thread and plays it through the system audio device.
"""

import time
import sounddevice as sd

from queue import Queue
from threading import Thread

from helpers import CONSOLE

from kokoro import KPipeline

# Voice selection mapping: human-friendly labels -> technical voice IDs
VOICE_LABEL_TO_ID = {
    "Seraphina Heart": "af_heart",
    "Arabella Vale": "af_bella",
    "Nicolette Thorne (recommended)": "af_nicole",
    "Emmeline Blackwood": "bf_emma",
    "Gideon Fenwick": "am_fenrir",
    "Michael Hawthorne": "am_michael",
    "Silas Puckett": "am_puck",
    "Akira Hoshino": "jf_alpha",
    "Ishani Kapoor": "hf_alpha",
}


def select_speaker_voice():
    """Interactively prompt the user to select a voice and return its voice ID.

    Returns:
        The technical voice identifier (e.g., 'bf_emma') for the selected voice.
    """
    CONSOLE.print("[bold]Select speaker voice:[/bold]")
    options = list(VOICE_LABEL_TO_ID.items())
    for idx, (label, vid) in enumerate(options, start=1):
        CONSOLE.print(f"[cyan]{idx}. {label}[/cyan] [grey50]({vid})[/grey50]")
    while True:
        choice = CONSOLE.input("[green]Voice #> [/green]").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][1]
        CONSOLE.print(f"[red]Please enter a number between 1 and {len(options)}[/red]")

class Speaker:
    """Background text-to-speech synthesizer and audio player.

    Uses Kokoro's KPipeline to synthesize audio from text on a worker thread and
    plays buffered audio with sounddevice. Text items are enqueued and rendered
    in order, allowing non-blocking generation and playback.
    """

    # Best Voices:
    # af_heart, af_bella, af_nicole, bf_emma
    # am_fenrir, am_michael, am_puck
    # jf_alpha, hf_alpha
    def __init__(self, voice="bf_emma", lang_code="b", sample_rate=24000):
        """Initialize the speaker with the chosen voice, language, and sample rate.

        Args:
            voice: Technical Kokoro voice ID to use for synthesis.
            lang_code: Language code passed to KPipeline (e.g., 'b').
            sample_rate: Output sampling rate in Hz for playback.
        """
        self._voice=voice
        self._sample_rate = sample_rate

        # Queues
        self._text_queue = Queue()
        self._audio_queue = Queue()

        # TTS pipeline
        self._pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')

        # Threading
        self._stop_processing_thread = False
        self._stop_play_thread = False

        self._play_thread = None
        self._processing_thread = None

    def _process(self):
        """Worker loop: consume text queue, synthesize audio, and enqueue audio chunks."""
        while not self._stop_processing_thread:
            if self._text_queue.empty():
                time.sleep(0.05)
                continue
            text = self._text_queue.get()

            try:
                generator = self._pipeline(text, voice=self._voice)
                for i, (gs, ps, audio) in enumerate(generator):
                    self._audio_queue.put(audio)
            except Exception as e:
                CONSOLE.print(f"[red]Error on audio synthesis {e}[/red]")

    def _play(self):
        """Worker loop: consume audio queue and play audio snippets sequentially."""
        while not self._stop_play_thread:
            if self._audio_queue.empty():
                time.sleep(0.05)
                continue
            audio_snippet = self._audio_queue.get()

            try:
                sd.play(audio_snippet, self._sample_rate)
                sd.wait()
            except Exception as e:
                CONSOLE.print(f"[red]Error on audio playing {e}[/red]")

    def add_text(self, message:str):
        """Enqueue a text message for synthesis and playback."""
        self._text_queue.put(message)

    def start(self):
        """Start the processing and playback worker threads if not already running."""
        self._stop_processing_thread = False
        self._stop_play_thread = False

        if self.is_running():
            return

        self._processing_thread = Thread(target=self._process, daemon=True)
        self._play_thread = Thread(target=self._play, daemon=True)

        if not self._processing_thread.is_alive():
            self._processing_thread.start()

        if not self._play_thread.is_alive():
            self._play_thread.start()

    def stop(self):
        """Signal both worker threads to stop on their next loop iteration."""
        self._stop_processing_thread = True
        self._stop_play_thread = True

    def stop_wait(self):
        """Stop the worker threads and block until they have fully terminated."""
        if not self.is_running():
            return

        self.stop()
        if self._processing_thread is not None:
            self._processing_thread.join()
        if self._play_thread is not None:
            self._play_thread.join()

    def is_running(self):
        """Return True if either the processing or playback worker thread is active."""
        processing_is_running = None is not self._processing_thread and self._processing_thread.is_alive()
        play_is_running = None is not self._play_thread and self._play_thread.is_alive()
        return processing_is_running or play_is_running
