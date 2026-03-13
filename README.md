## Ask Log

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://www.python.org/downloads/)
[![OS](https://img.shields.io/badge/OS-Windows%20|%20macOS%20|%20Linux-555)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Ask Log is a CLI-first assistant that helps you explore, summarize, and reason about application logs through a conversational interface. Point it at a log file, ask questions, and iterate quickly.

This project uses LangChain under the hood and supports multiple LLM providers (OpenAI, Anthropic, Google), configurable via a simple guided setup. Conversations are optionally saved so you can resume later.

---

### Quick links

[![Package](https://img.shields.io/badge/py-package-brightgreen?label=PyPI%20package&link=https%3A%2F%2Fpypi.org%2Fproject%2Fask-log%2F)](#-PYPI)

---

### Features

- Conversational log analysis over any text log
- Local vector index (FAISS) for retrieval-augmented answers
- Provider-agnostic via LangChain: OpenAI, Anthropic, Google Gemini
- Persistent conversation history per-log for continuity
- Colorful, ergonomic CLI using `rich` and `click`

---

## Installation

```bash
pip install ask-log
```

---

## Quickstart

Once installed, the `ask-log` command is available.

```bash
# 1) Configure your preferred provider and model
ask-log configure

# 2) Check configuration
ask-log status

# 3) Analyze a log file interactively; optionally save the conversation
ask-log chat --log-file /path/to/app.log --save ~/.ask-log/last-session.json
```

During configuration, you’ll be prompted for provider credentials and model. Supported providers are:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

---

## Commands

```bash
# Guided config flow (choose provider, model, and options)
ask-log configure

# Show current provider configuration
ask-log status

# Start an interactive session on a specific log file
ask-log chat --log-file /path/to/logfile.log --save path/to/convo.json

# Reset configuration (removes ~/.ask-log/config.yaml)
ask-log reset
```

Configuration is stored at `~/.ask-log/config.yaml`.

---

## Tips

- Use natural language questions like “What errors do you see?”, “Summarize main events”, “Any anomalies around 10:32?”
- Use `--save` to capture the conversation so you can resume context later.
- The first run on a large log may build a local vector index; subsequent runs will be faster.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, a suggested workflow, and development tips.

---

## License

## This project is licensed under the [MIT License](LICENSE).
