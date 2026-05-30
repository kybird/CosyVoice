"""Lightweight STT helper using faster-whisper.

Usage:
    python transcribe.py <audio_path> [language]

Prints transcribed text to stdout (one line).
Prints error messages to stderr and exits with code 1 on failure.
"""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: transcribe.py <audio_path> [language]", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "ko"

    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper not installed. Run: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)

    # Use tiny model for fast transcription (good enough for short reference prompts)
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio_path, language=language)
    text = "".join(s.text for s in segments).strip()

    print(text)


if __name__ == "__main__":
    main()
