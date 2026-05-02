#!/bin/bash
# Continuation mode example script

if [ $# -lt 1 ]; then
    echo "Usage: $0 <text>"
    echo "Example: $0 'Hello, this is a test of text to speech.'"
    exit 1
fi

TEXT="$1"

# Run continuation mode (models auto-download from HuggingFace)
./target/release/moss-tts-nano-rust-candle \
    --text "$TEXT" \
    --mode continuation \
    --output continuation_output.wav \
    --cpu \
    --do-sample \
    --audio-temperature 0.8 \
    --audio-top-p 0.95

echo "Output saved to: continuation_output.wav"