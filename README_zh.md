# moss-tts-nano-rust-candle

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%92_Model-ramishi%2Fmoss--tts--nano--candle-yellow.svg)](https://huggingface.co/ramishi/moss-tts-nano-candle)
[![English](https://img.shields.io/badge/English-README-blue.svg)](README.md)

一个基于 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 的 **Rust/Candle** 实现 — 来自 [MOSI.AI](https://mosi.cn/) 和 [OpenMOSS 团队](https://www.open-moss.com/) 的 0.1B 参数多语言文本转语音模型，支持语音克隆。

> **🙏 致谢：** 本项目是对 [OpenMOSS 团队](https://www.open-moss.com/) 和 [MOSI.AI](https://mosi.cn/) 杰出工作的独立 Rust 移植。他们开源的 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 模型 — 凭借其优雅的架构、高质量的语音合成和 Apache-2.0 许可证 — 使这个 Rust 实现成为可能。所有模型设计、训练和原始 Python 实现的荣誉都归属于他们。🌟

这是官方 Python/PyTorch 实现的**独立 Rust 移植版本**，基于 [Hugging Face Candle](https://github.com/huggingface/candle) 构建。它能够生成与 Python 参考实现数值完全一致的立体声 48 kHz 音频，且**无需 Python 运行时** — 只需一个编译好的二进制文件。

---

## 为什么选择 Rust？

与官方 Python 版 MOSS-TTS-Nano 对比：

| | Python/PyTorch | **Rust/Candle（本仓库）** |
|---|---|---|
| **运行时依赖** | Python 3.10+, PyTorch 2.7, torchaudio, transformers | 无（静态二进制文件） |
| **安装大小** | ~2 GB (PyTorch) | **~10 MB**（单个二进制文件） |
| **模型加载** | PyTorch `.bin` (BF16) | Safetensors (FP32, 直接 mmap) |
| **GPU 支持** | CUDA, MPS | CUDA, MPS (计划中) |
| **内存安全** | Python GC | **Rust 借用检查器** |
| **服务器部署** | FastAPI + uvicorn | axum/actix (Rust HTTP 生态) |
| **启动时间** | ~5s (Python import) | **~0.2s** (直接 mmap) |
| **交叉编译** | 复杂 | `cargo build --target` |
| **语音克隆** | 支持 | 支持 |
| **流式输出** | 支持 | 计划中 |

**在以下场景使用本移植版本：**
- 需要单个、无依赖的二进制文件，可在任何地方运行
- 需要最小内存占用和快速冷启动
- 想用 Rust 构建生产级 TTS API 服务器
- 想将 TTS 嵌入桌面/移动应用，无需 Python
- 需要通过 `Cargo.lock` 实现可复现构建

**在以下场景坚持使用官方 Python 版本：**
- 需要 GPU 加速（Candle 的 CUDA/MPS 支持仍在成熟中）
- 需要微调（训练代码仅支持 Python）
- 需要使用浏览器版 ONNX 版本（[MOSS-TTS-Nano-Reader](https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader)）

---

## 特性

- **语音克隆** — 使用参考音频片段的音色生成语音
- **续写模式** — 无需参考语音的纯文本转语音
- **长文本分块** — 自动句子分割，支持任意长度输入
- **HuggingFace 自动下载** — 首次运行时自动获取模型，本地缓存
- **离线模式** — `--local-only` 标志，支持隔离网络环境
- **完整采样控制** — 温度、top-p、top-k、重复惩罚、随机种子
- **立体声 48 kHz 输出** — Float32 WAV，与 Python 版本相同
- **验证一致性** — 与 Python 参考实现张量级数值匹配
- **零 Python 依赖** — 纯 Rust，静态链接

---

## 快速开始

### 前置要求

- [Rust 1.85+](https://rustup.rs/) (edition 2024)
- 首次运行需要网络连接（模型下载 ~373 MB）
- **仅 Linux：** `libsentencepiece-dev` C 库（`sudo apt-get install libsentencepiece-dev`）
- **macOS：** 通过 Homebrew 安装 `sentencepiece`（`brew install sentencepiece`）

### 构建和运行

```bash
# 克隆仓库
git clone https://github.com/ramishi/moss-tts-nano-rust-candle.git
cd moss-tts-nano-rust-candle

# Release 构建（优化，首次约 2 分钟）
cargo build --release

# 运行 — 模型会从 HuggingFace 自动下载
./target/release/moss-tts-nano-rust-candle \
  --text "Hello! This is MOSS, speaking to you from Rust." \
  --mode continuation \
  --cpu \
  --output hello.wav
```

首次运行会下载约 373 MB 的模型权重到 `~/.cache/huggingface/hub/`。后续运行会使用缓存。

### 语音克隆

```bash
./target/release/moss-tts-nano-rust-candle \
  --text "Today the weather is nice, perfect for a walk outside." \
  --mode voice_clone \
  --prompt-audio-path reference.wav \
  --cpu \
  --output cloned.wav
```

---

## CLI 参数参考

```
moss-tts-nano-rust-candle [OPTIONS]

必需的参数：
  -t, --text <TEXT>              要合成的文本

模式：
  -m, --mode <MODE>              合成模式："voice_clone" 或 "continuation" [默认: voice_clone]
      --prompt-audio-path <PATH>  voice_clone 模式的参考音频（voice_clone 模式必需）

输出：
  -o, --output <FILE>            输出 WAV 文件路径 [默认: output.wav]

模型路径（默认从 HuggingFace 缓存自动解析）：
      --tokenizer <PATH>         SentencePiece 分词器模型
      --config <PATH>            LM config.json
      --lm-weights <PATH>        LM safetensors 权重
      --codec-config <PATH>      音频分词器配置
      --codec-weights <PATH>     音频分词器 safetensors 权重
      --local-only               跳过 HuggingFace 下载，仅使用本地文件

采样参数：
      --do-sample / --no-do-sample   启用采样 [默认: true]
      --seed <N>                      随机种子 [默认: 40]
      --text-temperature <F>          文本 token 温度 [默认: 1.0]
      --text-top-p <F>                文本核采样阈值 [默认: 1.0]
      --text-top-k <N>                文本 top-k 采样 [默认: 50]
      --audio-temperature <F>         音频 token 温度 [默认: 1.0]
      --audio-top-p <F>               音频核采样阈值 [默认: 0.95]
      --audio-top-k <N>               音频 top-k 采样 [默认: 25]
      --audio-repetition-penalty <F>  音频重复惩罚 [默认: 1.2]
      --max-frames <N>                最大音频帧数 [默认: 375]

处理选项：
      --cpu                        强制使用 CPU 推理
      --normalize-text / --no-normalize-text  文本标准化 [默认: true]
      --voice-clone-max-text-tokens <N>       语音克隆每个块的最大文本 token 数 [默认: 35]
      --print-chunks               仅打印文本块并退出（voice_clone 模式）
      --export-latents <DIR>       导出中间张量用于调试（需要 debug-export feature）

示例：
  # 简单续写
  moss-tts-nano-rust-candle -t "Hello world" -m continuation --cpu

  # 语音克隆，自定义采样参数
  moss-tts-nano-rust-candle -t "你好世界" -m voice_clone --prompt-audio-path ref.wav \
    --audio-temperature 0.8 --audio-top-p 0.9 --seed 42 --cpu
```

---

## 架构

```
文本输入
    |
    v
+-------------+     +------------------+     +-----------------+
| 文本输入     |---->| SentencePiece    |---->| 文本标准化      |
| + 可选       |     | 分词器           |     | + 分块          |
| 参考音频     |     +------------------+     +--------+--------+
+------+-------+                                        |
       |                                               v
       |                                   +------------------+
       |                                   |   GPT-2 LM       |
       |                                   |  (0.1B 参数)     |
       |                                   |  自回归生成        |
       |                                   +--------+---------+
       |                                            |
       |              +------------------+          | 音频 token IDs
       +------------>|  音频分词器       |<---------+
       （编码参考）   |  (Encodec 风格)  |  （解码输出）
                     |  48kHz, 16 VQ    |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |  Float32 WAV     |
                     |  立体声 48kHz    |
                     +------------------+
```

**源代码布局：**

| 路径 | 用途 |
|------|---------|
| `src/main.rs` | CLI 入口点，HuggingFace 模型路径解析 |
| `src/pipeline.rs` | 顶层编排（voice_clone, text_to_speech） |
| `src/lib.rs` | 库入口 |
| `src/modules/attention.rs` | 注意力机制实现 |
| `src/modules/transformer.rs` | Transformer 模型实现 |
| `src/modules/gpt2.rs` | GPT-2 语言模型 |
| `src/modules/lfq.rs` | 有限标量量化 (LFQ) |
| `src/modules/rotary.rs` | 旋转位置编码 (RoPE) |
| `src/sampling.rs` | 采样策略（贪心、top-k、top-p、重复惩罚） |
| `src/testing.rs` | 测试用例 |

---

## 与官方 Python 版本的差异

| 方面 | 官方 Python 版 | 本 Rust 移植版 |
|------|----------------|----------------|
| 框架 | PyTorch | Candle |
| 模型权重 | `pytorch_model.bin` (BF16, 707 MB) | `moss_tts_nano_lm.safetensors` (FP32, 285 MB) |
| 音频分词器权重 | `pytorch_model.bin` 带 weight_norm 参数化 | 预合并的 safetensors（weight_norm 已分解） |
| 文本标准化 | WeTextProcessing (Python) | 自定义 Rust 实现（约 524 行） |
| 音频 I/O | torchaudio + soundfile | hound + symphonia + rubato |
| GPU 支持 | CUDA, MPS | 仅 CPU (CUDA/MPS 计划中) |
| 流式输出 | Server-sent events | 尚未实现 |
| API 服务器 | FastAPI (Python) | 尚未实现 |
| 微调 | 支持 | 不支持（请使用 Python 版） |
| ONNX 导出 | 支持 | 不适用（直接使用 safetensors） |

### 权重格式说明

官方 HuggingFace 仓库存储的音频分词器权重带有 `weight_norm` 参数化（每个权重分解为 `original0` x `original1`）。本 Rust 移植版使用预合并的权重，其中 weight_norm 已被撤销，为每个参数生成单个张量。合并后的权重经验证数值完全一致（所有 34 个量化器张量 0 个不匹配）。

本仓库中的 LM 权重是 FP32 safetensors (285 MB)，而不是官方的 BF16 pytorch_model.bin (707 MB)。两者包含相同的 194 个张量，数值相同，只是存储精度不同。

---

## 模型权重

模型权重托管在 HuggingFace 上，首次运行时自动下载：

| 文件 | 大小 | 描述 |
|------|------|-------------|
| `config.json` | 5 KB | LM 配置 |
| `tokenizer.model` | 460 KB | SentencePiece 分词器 |
| `moss_tts_nano_lm.safetensors` | 285 MB | 语言模型权重 (FP32) |
| `moss_audio_tokenizer_config.json` | 2 KB | 音频分词器配置 |
| `moss_audio_tokenizer.safetensors` | 88 MB | 音频分词器权重 (FP32, 已合并) |

**总计：~373 MB**（缓存在 `~/.cache/huggingface/hub/`）

**离线 / 隔离网络环境使用：**

```bash
# 预下载模型
./target/release/moss-tts-nano-rust-candle --text "warmup" -m continuation --cpu

# 然后在没有网络的情况下运行
./target/release/moss-tts-nano-rust-candle --text "offline text" -m continuation --local-only --cpu
```

或者指向明确的本地路径：
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

## 从源码构建

### 标准构建

```bash
cargo build --release
```

输出：`target/release/moss-tts-nano-rust-candle`（约 10 MB，已剥离，LTO 优化）

### 调试导出构建

如果需要 numpy 张量导出用于移植验证：

```bash
cargo build --release --features debug-export
```

这会启用 `--export-latents` 标志，并添加 `ndarray`/`ndarray-npy` 依赖。

### 要求

- Rust 1.85+ (edition 2024)
- C 编译器（用于 `sentencepiece-sys` 构建）
- **Linux：** `libsentencepiece-dev` (`sudo apt-get install libsentencepiece-dev`)
- **macOS：** 通过 Homebrew 安装 `sentencepiece` (`brew install sentencepiece`) — 通常由构建脚本自动解决
- 网络连接（用于 `cargo fetch` 和首次运行模型下载）

---

## 性能

在 macOS ARM64 (Apple M4, 仅 CPU) 上的基准测试：

| 模式 | TTFB | RTF | 输出 |
|------|------|-----|--------|
| 续写（英文） | ~180ms | ~0.07x | 2声道 48kHz |
| 语音克隆（中文） | ~475ms | ~0.17x | 2声道 48kHz |

- **TTFB** = 首个字节时间（生成第一个音频帧的时间）
- **RTF** = 实时因子（1.0 = 实时，越低越快）

内存使用：约 1.5 GB RAM（模型权重 + 推理缓冲区）

---

## 路线图

- [ ] **流式音频输出** — 在生成时逐步输出音频帧，而不是等待完整生成
- [ ] **HTTP API 服务器** — 基于 axum 的 REST API，支持 `/synthesize` 和 `/synthesize/stream` 端点
- [ ] **CUDA 支持** — 通过 Candle CUDA 后端进行 GPU 推理
- [ ] **MPS 支持** — 通过 Metal Performance Shaders 支持 Apple Silicon GPU
- [ ] **量化推理** — INT8/FP16 权重，减少内存占用
- [ ] **SSML 支持** — 语音合成标记语言，用于细粒度控制
- [ ] **批量推理** — 并发处理多个请求
- [ ] **Docker 镜像** — 预构建的部署容器
- [ ] **Python 绑定** — PyO3 包装器，可从 Python 使用
- [ ] **基准测试套件** — 跨后端的系统性能比较

---

## 贡献

欢迎贡献！这是一个独立的社区移植项目，我们感谢任何帮助。

### 入门指南

```bash
# Fork 并克隆
git clone https://github.com/YOUR_USERNAME/moss-tts-nano-rust-candle.git
cd moss-tts-nano-rust-candle

# 构建
cargo build

# 运行测试
cargo test

# 代码检查
cargo clippy -- -D warnings

# 格式检查
cargo fmt --check
```

### 代码风格

- 提交前运行 `cargo fmt`
- 解决所有 `cargo clippy` 警告
- 为新功能添加测试
- 保持公共 API 精简 — 大多数代码是 pipeline 内部的

### 添加功能

1. 开一个 issue 描述功能和动机
2. 保持 PR 专注 — 每个 PR 一个功能
3. 确保 `cargo test` 和 `cargo clippy` 通过
4. 如果功能改变了用户可见行为，更新 README.md

### 报告 Bug

提交 bug 时，请包含：
- Rust 版本 (`rustc --version`)
- 操作系统和架构
- 使用的完整命令行
- 完整的错误输出
- 如果可能，提供最小复现案例

---

## 致谢

本项目是 [OpenMOSS 团队](https://www.open-moss.com/) 和 [MOSI.AI](https://mosi.cn/) 的 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 的 Rust 移植版。

**原作者：** Yitian Gong, Botian Jiang, Yiwei Zhao, Yucheng Yuan, Kuangwei Chen, Yaozhou Jiang, Cheng Chang, Dong Hong, Mingshu Chen, Ruixiao Li, Yiyang Zhang, Yang Gao, Hanfu Chen, Ke Chen, Songlin Wang, Xiaogui Yang, Yuqian Zhang, Kexin Huang, ZhengYuan Lin, Kang Yu, Ziqi Chen, Jin Wang, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu

**Rust 移植者：** [Simon Law (ramishi)](https://github.com/ramishi)

本移植版得以实现离不开：
- [Candle](https://github.com/huggingface/candle) — Hugging Face Rust ML 框架
- [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) — 原始模型和 Python 实现
- [OpenMOSS Team](https://www.open-moss.com/) — 在 Apache-2.0 下开源模型

---

## 许可证

根据 [Apache License, Version 2.0](LICENSE) 许可证授权。

本项目是 MOSS-TTS-Nano 的衍生作品，后者同样根据 Apache-2.0 许可证授权。

---

## 引用

如果您使用本工作，请引用原始 MOSS-TTS 论文：

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
