# VaultRAG

A powerful offline-capable chatbot with **Retrieval-Augmented Generation (RAG)** that lets you chat with AI using local knowledge bases like Wikipedia, Python documentation, or any ZIM file archive.

> **Note:** This project runs entirely locally using `llama-cpp-python` and GGUF models. No internet is required after initial setup.

---

## Quick Setup

### Prerequisites

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Linux (tested on Ubuntu/Debian)                                           │
│ • NVIDIA GPU (Recommended) - RTX 3060 or better for optimal speed           │
│ • 12GB+ RAM (System) + 8GB+ VRAM (GPU)                                      │
│ • Python 3.8+                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Installation

**Step 1: Run the setup script**
```bash
chmod +x setup.sh
./setup.sh
```

> **What the setup script does:**
> - Installs system dependencies (Python, libzim)
> - Sets up a virtual environment
> - Installs PyTorch with CUDA support (for GPU acceleration)
> - Installs `llama-cpp-python` for local GGUF inference
> - Creates a `krag` command system-wide

**Step 2: Add Models and Data**
- **Models**: The system will automatically download the required GGUF models (DarkIdol 8B, Aletheia 3B) to the `shared_models/` directory on first run.
- **Data (ZIM)**: Download a ZIM file (e.g., from [Kiwix](https://library.kiwix.org/)) and place it in the project root. (e.g., `wikipedia_en_all_maxi_2025-08.zim`).

**Step 3: Start the chatbot**
```bash
vrag
```

### Uninstallation

To remove the application, use the included GUI uninstaller:
```bash
./uninstall.sh
```
This tool allows you to safely remove the virtual environment, models, and cache **while strictly protecting your ZIM files**.

---

## Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Local AI: Runs GGUF models (DarkIdol 8B) locally                          │
│ • RAG: Retrieves facts from ZIM files                                       │
│ • Multi-Joint Architecture: Uses specialized small models for reasoning     │
│ • Just-In-Time Indexing: No long pre-indexing wait; indexes on-the-fly      │
│ • GPU Accelerated: Uses NVIDIA GPU for both Embeddings and LLM Generation   │
│ • Privacy Focused: 100% Offline (after setup)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How It Works

VaultRAG uses a **Local Multi-Joint Architecture**:

1.  **Joint 1 (Entity extraction)**: Analyzes your query to find key terms.
2.  **Retrieval**: Searches the ZIM file using a hybrid of Keyword (BM25-like) and Semantic (Embeddings) search.
3.  **Joint 2 (Scoring)**: Reads candidate articles and scores them for relevance.
4.  **Joint 3 (Filtering)**: Extracts the exact paragraphs containing the answer.
5.  **Generation**: The main Chat Model (DarkIdol 8B) synthesizes the answer from the retrieved facts.

**VRAM Optimization**: To run on consumer GPUs (e.g. 12GB VRAM), the system intelligently loads and unloads models as needed to prevent Out-Of-Memory errors.

## Configuration

### Models
Models are defined in `chatbot/config.py`.
- **Default Chat Model**: `Ishaanlol/.../Aletheia-3B`
- **RAG/Joint Model**: `Ishaanlol/.../Aletheia-3B`

You can change these to any Hugging Face repo ID containing GGUF files.

## Troubleshooting

### "Failed to create llama_context" (OOM)
This means you ran out of VRAM. The system now has a protections against this, but if you see it, ensure no other GPU-heavy apps are running.

### "CUDA not available"
If `vrag` says it's using CPU:
1.  Ensure you have NVIDIA drivers installed.
2.  Re-run `./setup.sh` to reinstall PyTorch.

### "Dependencies missing"
Run `./setup.sh` again to fix any broken python packages.

---

**License**: This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.
