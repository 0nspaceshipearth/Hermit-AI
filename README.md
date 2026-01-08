# Hermit — Offline AI Chatbot for Wikipedia & ZIM Files

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Accelerated-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

>  **A privacy-first, offline AI chatbot** powered by local LLMs and Retrieval-Augmented Generation (RAG). Chat with Wikipedia, documentation, or any ZIM archive — completely offline, 100% private.

**No cloud. No API keys. No data leaves your machine.**

---

##  Key Features

| Feature | Description |
|---------|-------------|
| 🔒 **100% Offline** | Runs entirely on your local machine after initial setup |
| 🧠 **Local LLM** | Uses GGUF models via `llama-cpp-python` — no OpenAI needed |
| 📚 **Wikipedia RAG** | Chat with offline Wikipedia using [Kiwix ZIM files](https://library.kiwix.org/) |
| ⚡ **GPU Accelerated** | CUDA support for fast inference on NVIDIA GPUs |
| 🎯 **Multi-Joint Architecture** | Unique 3-stage reasoning pipeline for accurate answers |
| 🔍 **Hybrid Search** | Combines keyword (BM25) + semantic (FAISS) retrieval |
| 🔨 **Forge ZIM Creator** | Build your own knowledge bases from PDF, DOCX, TXT, Markdown |
| 🛡️ **Privacy First** | Your data never leaves your computer |

---

## Quick Start

### Prerequisites

- **OS**: Linux (Ubuntu/Debian tested)
- **GPU**: NVIDIA RTX 3060+ recommended (8GB+ VRAM)
- **RAM**: 12GB+ system memory
- **Python**: 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/imDelivered/Hermit-AI.git
cd Hermit-AI

# Run the setup script
chmod +x setup.sh
./setup.sh
```

> **What setup.sh does:**
> - Installs system dependencies (Python, libzim, CUDA toolkit)
> - Creates an isolated virtual environment
> - Installs PyTorch with CUDA 12.1 support
> - Compiles `llama-cpp-python` with GPU acceleration
> - Creates the `hermit` command system-wide

### Add Your Knowledge Base

Download a ZIM file from [Kiwix Library](https://library.kiwix.org/) and place it in the project root:

```bash
# Example: Download Wikipedia
wget https://download.kiwix.org/zim/wikipedia_en_all_maxi.zim
```

### Launch Hermit

```bash
hermit              # Start the GUI
hermit --cli        # Start in terminal mode
hermit --debug      # Start with verbose logging
```

### Create Your Own ZIM (Forge)

Use **Forge** to create custom knowledge bases from your documents:

```bash
forge               # Launch Forge GUI
forge /path/to/docs -o myknowledge.zim  # CLI mode
```

**Supported formats:** TXT, Markdown, PDF, DOCX, HTML, EPUB

---

## How It Works — Multi-Joint RAG Architecture

Hermit uses a unique **Multi-Joint Architecture** that chains specialized reasoning stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│  User Query: "How did the Roman Empire fall?"                       │
├─────────────────────────────────────────────────────────────────────┤
│  [Joint 1] Entity Extraction                                        │
│     → Extracts: "Roman Empire", "fall", "decline"                  │
├─────────────────────────────────────────────────────────────────────┤
│  [Retrieval] Hybrid Search (BM25 + FAISS)                           │
│     → Finds 15 candidate articles from ZIM file                    │
├─────────────────────────────────────────────────────────────────────┤
│  [Joint 2] Article Scoring                                          │
│     → Scores articles 0-10, selects top 5                          │
├─────────────────────────────────────────────────────────────────────┤
│  [Joint 3] Chunk Filtering                                          │
│     → Extracts most relevant paragraphs                            │
├─────────────────────────────────────────────────────────────────────┤
│  [Generation] Final Answer                                          │
│     → LLM synthesizes answer from verified facts                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Why Multi-Joint?**
- ✅ Reduces hallucinations by grounding in retrieved facts
- ✅ Uses specialized small models for each reasoning step
- ✅ GBNF grammar enforcement ensures valid JSON at every stage
- ✅ Just-in-time indexing — no pre-processing wait

---

## ⚙️ Configuration

### Models

Edit `chatbot/config.py` to customize:

```python
# Default model (auto-downloads from Hugging Face)
DEFAULT_MODEL = "Ishaanlol/Aletheia-Llama-3.2-3B"

# Joint models (entity extraction, scoring, filtering)
ENTITY_JOINT_MODEL = DEFAULT_MODEL
SCORER_JOINT_MODEL = DEFAULT_MODEL
FILTER_JOINT_MODEL = DEFAULT_MODEL

# Context window size
DEFAULT_CONTEXT_SIZE = 16384
```

### Supported Models

Any GGUF model from Hugging Face works. Recommended:
- **Aletheia 3B** (default) — Fast, accurate
- **Llama 3.2 3B** — Great reasoning
- **Mistral 7B** — More capable, needs 12GB+ VRAM

---

## Troubleshooting

### "Failed to create llama_context" (Out of Memory)
Your GPU ran out of VRAM. Solutions:
1. Close other GPU applications
2. Use a smaller model
3. Reduce `DEFAULT_CONTEXT_SIZE` in config

### "CUDA not available"
If Hermit uses CPU instead of GPU:
1. Ensure NVIDIA drivers are installed: `nvidia-smi`
2. Re-run `./setup.sh` to reinstall PyTorch with CUDA

### "Dependencies missing"
```bash
./setup.sh  # Re-run to fix broken packages
```

---

## 🗑️ Uninstallation

```bash
./uninstall.sh
```

The GUI uninstaller lets you selectively remove:
- ✅ Virtual environment
- ✅ Downloaded models
- ✅ Search indexes
- 🛡️ **Your ZIM files are always protected**

---

## 🤝 Contributing

Contributions welcome! Please read the codebase and open a PR.

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See [LICENSE](LICENSE) for details.

---

## ⭐ Star History

If you find Hermit useful, please give it a star! It helps others discover the project.

---

<p align="center">
  <b>Hermit</b> — Your offline AI companion 🧙‍♂️
</p>
