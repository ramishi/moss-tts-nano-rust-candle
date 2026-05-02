# moss-tts-nano-rust-candle

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Model-ramishi%2Fmoss--tts--nano--candle-yellow.svg)](https://huggingface.co/ramishi/moss-tts-nano-candle)

A **Rust/Candle** implementation of [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) — a 0.1B parameter multilingual text-to-speech model with voice cloning, from [MOSI.AI](https://mosi.cn/) and the [OpenMOSS team](https://www.open-moss.com/).

This is an **independent Rust port** of the official Python/PyTorch implementation, built on [Hugging Face Candle](https://github.com/huggingface/candle). It produces stereo 48 kHz audio with verified numerical parity against the Python reference, and requires **no Python runtime** — just a single compiled binary.

---

## Why Rust?

Compare to the official python MOSS-TTS-Nano :

| | Python/PyTorch |  **Rust/Candle (this repo)** |
|---|---|---|---|
| **Runtime deps** | Python 3.10+, PyTorch 2.7, torchaudio, transformers | None (static binary) |
| **Install size** | ~2 GB (PyTorch) | **~10 MB** (single binary) |
| **Model loading** | PyTorch `.bin` (BF16) | Safetensors (FP32, direct mmap) |
| **GPU support** | CUDA, MPS | CUDA, MPS (planned) |
| **Memory safety** | Python GC | **Rust borrow checker** |
| **Server deployment** | FastAPI + uvicorn | axum/actix (Rust HTTP ecosystem) |
| **Startup time** | ~5s (Python import) | **~0.2s** (direct mmap) |
| **Cross-compile** | Complex | `cargo build --target` |
| **Voice cloning** | Yes | Yes |
| **Streaming** | Yes | Planned |

**Use this port when you want:**
- A single, dependency-free binary that runs anywhere
- Minimal memory footprint and fast cold start
- To build a production TTS API server in Rust
- To embed TTS into a desktop/mobile app without Python
- Reproducible builds via `Cargo.lock`

**Stick with the official Python version when you need:**
- GPU acceleration (CUDA/MPS support in Candle is still maturing)
- Fine-tuning (training code is Python-only)
- The browser-based ONNX version ([MOSS-TTS-Nano-Reader](https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader))

---

## Features

- **Voice cloning** — generate speech in the timbre of a reference audio clip
- **Continuation mode** — pure text-to-speech without a reference voice
- **Long text chunking** — automatic sentence splitting for arbitrary-length input
- **HuggingFace auto-download** — models fetched on first run, cached locally
- **Offline mode** — `--local-only` flag for air-gapped environments
- **Full sampling control** — temperature, top-p, top-k, repetition penalty, seed
- **Stereo 48 kHz output** — Float32 WAV, the same as the Python version
- **Verified parity** — tensor-level numerical match with Python reference
- **Zero Python dependency** — pure Rust with static linking

---

## Quick Start

### Prerequisites

- [Rust 1.85+](https://rustup.rs/) (edition 2024)
- Internet connection for first run (model download ~373 MB)
- **Linux only:** `libsentencepiece-dev` C library (`sudo apt-get install libsentencepiece-dev` on Debian/Ubuntu)

### Build and Run

```bash
# Clone
git clone https://github.com/ramishi/moss-tts-nano-rust-candle.git
cd moss-tts-nano-rust-candle

# Release build (optimized, ~2 min first time)
cargo build --release

# Run — models are downloaded automatically from HuggingFace
./target/release/moss-tts-nano-rust-candle \
  --text "Hello! This is MOSS, speaking to you from Rust." \
  --mode continuation \
  --cpu \
  --output hello.wav
```

First run downloads ~356 MB of model weights to `~/.cache/huggingface/hub/`. Subsequent runs use the cache.

### Voice Cloning

```bash
./target/release/moss-tts-nano-rust-candle \
  --text "Today the weather is nice, perfect for a walk outside." \
  --mode voice_clone \
  --prompt-audio-path reference.wav \
  --cpu \
  --output cloned.wav
```

---

## CLI Reference

```
moss-tts-nano-rust-candle [OPTIONS]

Required:
  -t, --text <TEXT>              Text to synthesize

Mode:
  -m, --mode <MODE>              Synthesis mode: "voice_clone" or "continuation" [default: voice_clone]
      --prompt-audio-path <PATH> Reference audio for voice_clone mode (required for voice_clone)

Output:
  -o, --output <FILE>            Output WAV file path [default: output.wav]

Model paths (auto-resolved from HuggingFace cache by default):
      --tokenizer <PATH>         SentencePiece tokenizer model
      --config <PATH>            LM config.json
      --lm-weights <PATH>        LM safetensors weights
      --codec-config <PATH>      Audio tokenizer config
      --codec-weights <PATH>     Audio tokenizer safetensors weights
      --local-only               Skip HuggingFace download, use local files only

Sampling:
      --do-sample / --no-do-sample   Enable sampling [default: true]
      --seed <N>                      Random seed [default: 40]
      --text-temperature <F>          Text token temperature [default: 1.0]
      --text-top-p <F>                Text nucleus sampling threshold [default: 1.0]
      --text-top-k <N>                Text top-k sampling [default: 50]
      --audio-temperature <F>         Audio token temperature [default: 1.0]
      --audio-top-p <F>               Audio nucleus sampling threshold [default: 0.95]
      --audio-top-k <N>               Audio top-k sampling [default: 25]
      --audio-repetition-penalty <F>  Audio repetition penalty [default: 1.2]
      --max-frames <N>                Maximum audio frames to generate [default: 375]

Processing:
      --cpu                        Force CPU inference
      --normalize-text / --no-normalize-text  Text normalization [default: true]
      --voice-clone-max-text-tokens <N>       Max text tokens per voice-clone chunk [default: 35]
      --print-chunks               Print text chunks and exit (voice_clone mode)
      --export-latents <DIR>       Export intermediate tensors for debugging (requires debug-export feature)

Examples:
  # Simple continuation
  moss-tts-nano-rust-candle -t "Hello world" -m continuation --cpu

  # Voice clone with custom sampling
  moss-tts-nano-rust-candle -t "你好世界" -m voice_clone --prompt-audio-path ref.wav \
    --audio-temperature 0.8 --audio-top-p 0.9 --seed 42 --cpu
```

---

## Architecture

```
Text Input
    |
    v
+-------------+     +------------------+     +-----------------+
| Text Input   |---->| SentencePiece    |---->| Text Normalize  |
| + Optional   |     | Tokenizer        |     | + Chunking      |
| Ref Audio    |     +------------------+     +--------+--------+
+------+-------+                                        |
       |                                               v
       |                                   +------------------+
       |                                   |   GPT-2 LM       |
       |                                   |  (0.1B params)   |
       |                                   |  Autoregressive   |
       |                                   +--------+---------+
       |                                            |
       |              +------------------+          | audio token IDs
       +------------>|  Audio Tokenizer |<---------+
       (encode ref)  |  (Encodec-style) |  (decode output)
                     |  48kHz, 16 VQ    |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |  Float32 WAV     |
                     |  Stereo 48kHz    |
                     +------------------+
```

**Source layout:**

| Path | Purpose |
|------|---------|
| `src/main.rs` | CLI entry point, HuggingFace model resolution |
| `src/pipeline.rs` | Top-level orchestration (voice_clone, text_to_speech) |
| `src/models/lm.rs` | Language model (GPT-2 autoregressive generation) |
| `src/models/audio_tokenizer.rs` | Audio codec (encode audio to tokens, decode tokens to waveform) |
| `src/models/input_builder.rs` | Build model input from text + audio tokens |
| `src/models/text_normalize.rs` | Chinese/English text normalization + sentence chunking |
| `src/models/tokenizer.rs` | SentencePiece wrapper |
| `src/models/audio_io.rs` | WAV/MP3/FLAC reading + resampling (rubato/symphonia) |
| `src/models/prompting.rs` | Voice clone prompt construction |
| `src/modules/` | Neural network primitives (attention, transformer, LFQ, rotary, etc.) |
| `src/sampling.rs` | Sampling strategies (greedy, top-k, top-p, repetition penalty) |

---

## Differences from Official Python Version

| Aspect | Official Python | This Rust Port |
|--------|----------------|----------------|
| Framework | PyTorch | Candle |
| Model weights | `pytorch_model.bin` (BF16, 707 MB) | `moss_tts_nano_lm.safetensors` (FP32, 285 MB) |
| Audio tokenizer weights | `pytorch_model.bin` with weight_norm parametrization | Pre-merged safetensors (weight_norm decomposed) |
| Text normalization | WeTextProcessing (Python) | Custom Rust implementation (~524 lines) |
| Audio I/O | torchaudio + soundfile | hound + symphonia + rubato |
| GPU support | CUDA, MPS | CPU only (CUDA/MPS planned) |
| Streaming output | Server-sent events | Not yet implemented |
| API server | FastAPI (Python) | Not yet implemented |
| Fine-tuning | Supported | Not supported (use Python) |
| ONNX export | Supported | N/A (uses safetensors directly) |

### Weight Format Notes

The official HuggingFace repos store the audio tokenizer weights with `weight_norm` parametrization (each weight is decomposed into `original0` x `original1`). This Rust port uses pre-merged weights where the weight_norm has been undone, producing a single tensor per parameter. The merged weights are verified to be numerically identical (0 mismatches across all 34 quantizer tensors).

The LM weights in this repo are FP32 safetensors (285 MB) rather than the official BF16 pytorch_model.bin (707 MB). Both contain the same 194 tensors with identical values, just stored at different precision.

---

## Model Weights

Model weights are hosted on HuggingFace and downloaded automatically:

| File | Size | Description |
|------|------|-------------|
| `config.json` | 5 KB | LM configuration |
| `tokenizer.model` | 460 KB | SentencePiece tokenizer |
| `moss_tts_nano_lm.safetensors` | 285 MB | Language model weights (FP32) |
| `moss_audio_tokenizer_config.json` | 2 KB | Audio tokenizer configuration |
| `moss_audio_tokenizer.safetensors` | 88 MB | Audio tokenizer weights (FP32, merged) |

**Total: ~373 MB** (cached in `~/.cache/huggingface/hub/`)

**Offline / air-gapped use:**

```bash
# Pre-download models
./target/release/moss-tts-nano-rust-candle --text "warmup" -m continuation --cpu

# Then run without network
./target/release/moss-tts-nano-rust-candle --text "offline text" -m continuation --local-only --cpu
```

Or point to explicit local paths:
```bash
./target/release/moss-tts-nano-rust-candle \
  --text "hello" -m continuation --cpu \
  --tokenizer /path/to/tokenizer.model \
  --config /path/to/config.json \
  --lm-weights /path/to/moss_tts_nano_lm.safetensors \
  --codec-config /path/to/moss_audio_tokenizer_config.json \
  --codec-weights /path/to/moss_audio_tokenizer.safetensors
```

---

## Building from Source

### Standard Build

```bash
cargo build --release
```

Output: `target/release/moss-tts-nano-rust-candle` (~10 MB, stripped, LTO-optimized)

### Debug Export Build

If you need numpy tensor export for porting validation:

```bash
cargo build --release --features debug-export
```

This enables the `--export-latents` flag and adds `ndarray`/`ndarray-npy` dependencies.

### Requirements

- Rust 1.85+ (edition 2024)
- C compiler (for `sentencepiece-sys` build)
- **Linux:** `libsentencepiece-dev` (`sudo apt-get install libsentencepiece-dev`)
- **macOS:** `sentencepiece` via Homebrew (`brew install sentencepiece`) — usually resolved automatically by the build script
- Internet connection (for `cargo fetch` and first-run model download)

---

## Performance

Benchmarks on macOS ARM64 (Apple M4, CPU only):

| Mode | TTFB | RTF | Output |
|------|------|-----|--------|
| Continuation (English) | ~180ms | ~0.07x | 2ch 48kHz |
| Voice Clone (Chinese) | ~475ms | ~0.17x | 2ch 48kHz |

- **TTFB** = Time To First Byte (first audio frame generated)
- **RTF** = Real-Time Factor (1.0 = real-time, lower is faster)

Memory usage: ~1.5 GB RAM (model weights + inference buffers)

---

## Roadmap

- [ ] **Streaming audio output** — yield audio frames as they are generated, instead of waiting for full generation
- [ ] **HTTP API server** — axum-based REST API with `/synthesize` and `/synthesize/stream` endpoints
- [ ] **CUDA support** — GPU inference via Candle CUDA backend
- [ ] **MPS support** — Apple Silicon GPU via Metal Performance Shaders
- [ ] **Quantized inference** — INT8/FP16 weights for reduced memory footprint
- [ ] **SSML support** — Speech Synthesis Markup Language for fine-grained control
- [ ] **Batch inference** — Process multiple requests concurrently
- [ ] **Docker image** — Pre-built container for deployment
- [ ] **Python bindings** — PyO3 wrapper for use from Python
- [ ] **Benchmark suite** — Systematic performance comparison across backends

---

## Contributing

Contributions are welcome! This is an independent community port and we appreciate help.

### Getting Started

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/moss-tts-nano-rust-candle.git
cd moss-tts-nano-rust-candle

# Build
cargo build

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt --check
```

### Code Style

- Run `cargo fmt` before committing
- Resolve all `cargo clippy` warnings
- Add tests for new functionality
- Keep the public API minimal — most code is internal to the pipeline

### Adding a Feature

1. Open an issue describing the feature and its motivation
2. Keep PRs focused — one feature per PR
3. Ensure `cargo test` and `cargo clippy` pass
4. Update README.md if the feature changes user-facing behavior

### Reporting Bugs

When filing a bug, please include:
- Rust version (`rustc --version`)
- OS and architecture
- Full command line used
- Complete error output
- If possible, a minimal reproduction case

---

## Credits

This project is a Rust port of [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) by the [OpenMOSS Team](https://www.open-moss.com/) and [MOSI.AI](https://mosi.cn/).

**Original authors:** Yitian Gong, Botian Jiang, Yiwei Zhao, Yucheng Yuan, Kuangwei Chen, Yaozhou Jiang, Cheng Chang, Dong Hong, Mingshu Chen, Ruixiao Li, Yiyang Zhang, Yang Gao, Hanfu Chen, Ke Chen, Songlin Wang, Xiaogui Yang, Yuqian Zhang, Kexin Huang, ZhengYuan Lin, Kang Yu, Ziqi Chen, Jin Wang, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu

**Rust port by:** [Simon Law (ramishi)](https://github.com/ramishi)

This port would not be possible without:
- [Candle](https://github.com/huggingface/candle) — Hugging Face Rust ML framework
- [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) — The original model and Python implementation
- [OpenMOSS Team](https://www.open-moss.com/) — For open-sourcing the model under Apache-2.0

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

This project is a derivative work of MOSS-TTS-Nano, which is also licensed under Apache-2.0.

---

## Citation

If you use this work, please cite the original MOSS-TTS paper:

```bibtex
@misc{gong2026mossttstechnicalreport,
  title={MOSS-TTS Technical Report},
  author={Yitian Gong and Botian Jiang and Yiwei Zhao and Yucheng Yuan and Kuangwei Chen and Yaozhou Jiang and Cheng Chang and Dong Hong and Mingshu Chen and Ruixiao Li and Yiyang Zhang and Yang Gao and Hanfu Chen and Ke Chen and Songlin Wang and Xiaogui Yang and Yuqian Zhang and Kexin Huang and ZhengYuan Lin and Kang Yu and Ziqi Chen and Jin Wang and Zhaoye Fei and Qinyuan Cheng and Shimin Li and Xipeng Qiu},
  year={2026},
  eprint={2603.18090},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2603.18090}
}
```

```bibtex
@misc{openmoss2026mossttsnano,
  title={MOSS-TTS-Nano},
  author={OpenMOSS Team},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/OpenMOSS/MOSS-TTS-Nano}
}
```
