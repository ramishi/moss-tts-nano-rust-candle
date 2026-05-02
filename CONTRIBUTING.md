# Contributing to moss-tts-nano-rust-candle

Thanks for your interest in contributing! This is an independent community port of MOSS-TTS-Nano, and we welcome improvements.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/moss-tts-nano-rust-candle.git
cd moss-tts-nano-rust-candle

# Build
cargo build

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Code Style

- **Formatting**: Run `cargo fmt` before every commit
- **Linting**: All `cargo clippy` warnings must be resolved
- **Documentation**: Public functions must have doc comments
- **Tests**: New features should include tests where practical

## Pull Request Process

1. **Open an issue first** — Describe the change and its motivation
2. **One feature per PR** — Keep pull requests focused
3. **Pass CI** — `cargo test` and `cargo clippy` must pass
4. **Update documentation** — If user-facing, update README.md

## Reporting Bugs

When filing a bug report, please include:

- Rust version: `rustc --version`
- OS and architecture
- Complete command line used
- Full error output (no truncation)
- Minimal reproduction case if possible

## Feature Requests

Open an issue with:
- Clear description of the feature
- Motivation: why is this useful?
- Possible implementation approach (optional)

## Code of Conduct

Be respectful and constructive. This is a small community project — treat others as you would want to be treated.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license that covers this project.
