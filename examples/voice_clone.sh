#!/bin/bash
# Voice Clone example script

if [ $# -lt 2 ]; then
    echo "Usage: $0 <reference_audio.wav> <text>"
    echo "Example: $0 speaker.wav 'Hello, this is a test.'"
    exit 1
fi

REFERENCE_AUDIO="$1"
TEXT="$2"

# Run voice clone mode (models auto-download from HuggingFace)
./target/release/moss-tts-nano-rust-candle \
    --text "$TEXT" \
    --prompt-audio-path "$REFERENCE_AUDIO" \
    --mode voice_clone \
    --output voice_clone_output.wav \
    --cpu \
    --do-sample \
    --audio-temperature 0.8 \
    --audio-top-p 0.95

echo "Output saved to: voice_clone_output.wav"