# Hearth

> **Fully offline, privacy-first AI coding assistant and project agent for local Ollama models.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20with-uv-black)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Offline First](https://img.shields.io/badge/100%25-Offline%20%26%20Private-purple.svg)](#)

## Overview

Hearth is a local AI coding assistant that runs entirely on your machine. Powered by [Ollama](https://ollama.com/), it combines fast hybrid retrieval-augmented generation (RAG) with a strict, human-in-the-loop safety model. 

You get the power of an agentic coding assistant—codebase Q&A, multi-file edits, test running, and git operations—without your code ever leaving your device.

## Core Philosophy

- 🔒 **100% Offline & Private:** No telemetry, no cloud APIs, no remote model fallbacks. Your codebase stays on your machine.
- 🛡️ **Safety-First Tool Use:** Default-deny policy engine, strict path jailing, and mandatory human approval for any side effects (edits, shell commands, git writes).
- 🧠 **Hybrid RAG:** Fast, accurate codebase understanding using lexical (FTS5) and dense vector embeddings.
- 💻 **Local-First Hardware:** Optimized for consumer GPUs (e.g., 6GB+ VRAM laptops) and Apple Silicon, with tiered model recommendations.

## 🚧 Current Status

Hearth is currently in **active development**, building towards an MVP ("Private Ask + Supervised Edits"). 

If you are interested in the architecture, the project is heavily documented:
- 📐 [System Design & Data Model](docs/system-design.md)
- 🛡️ [Safety & Tool Use Policy](docs/safety-and-tool-use.md)
- 🗺️ [Implementation Roadmap](docs/implementation-roadmap.md)
- 🏗️ [Project Structure](docs/project-structure.md)
- 🤖 [Model Recommendations](docs/model-recommendations.md)

## 🛠️ Getting Started (Preview)

*Note: Hearth is not yet ready for daily use. This is a preview of the planned setup.*

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

### Installation
```bash
git clone https://github.com/<your-username>/hearth.git
cd hearth
uv sync
uv run hearth doctor
