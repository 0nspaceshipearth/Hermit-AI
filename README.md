# Hermit: offline ai chatbot for zim files

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Accelerated-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**hermit** is a privacy-first, 100% offline ai chatbot that lets you chat with wikipedia or any other .zim archive. no cloud, no api keys, no tracking. everything stays on your machine.

---
<img width="895" height="697" alt="hermit screenshot" src="https://github.com/user-attachments/assets/a60de92a-38cf-42a8-bd31-ca96429d5bf5" />


### why?
i wanted a way to search massive knowledge bases like wikipedia without needing an internet connection. hermit uses a "multi-joint" rag setup to make sure it actually reads the articles before answering, which cuts down on hallucinations a lot.

### features
- **100% local**: runs via `llama-cpp-python` — you don't even need ollama installed.
- **wikipedia rag**: search and chat with any [kiwix zim file](https://library.kiwix.org/).
- **gpu fast**: supports cuda for rtx cards so it's snappy.
- **smart retrieval**: uses a multi-stage pipeline to extract entities and score articles before answering.
- **forge**: tool to turn your own pdfs/docs into zim files for hermit to read.

---

### quick start

**prerequisites:**
- linux (ubuntu/debian works best)
- python 3.8+
- nvidia gpu recommended (rtx 3060+ / 8gb+ vram)
- 12gb+ system ram

```bash
# clone and setup
git clone https://github.com/imDelivered/Hermit-AI.git
cd Hermit-AI
chmod +x setup.sh
./setup.sh
```

the setup script handles the venv, torch, and builds `llama-cpp-python` with cuda support.

### how to use

grab a .zim file from the [kiwix library](https://library.kiwix.org/) and drop it in the project folder.

```bash
hermit        # starts the gui
hermit --cli  # run in your terminal
hermit --debug # verbose logging
```

---

### how it works (the multi-joint pipeline)

instead of just doing a basic vector search, hermit chains a few small models together:

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

this adds latency but makes the answers way more reliable.

---

### forge: create your own zim files

use **forge** to build custom knowledge bases from your own documents:

```bash
forge               # launch forge gui
forge /path/to/docs -o myknowledge.zim  # cli mode
```

**supported formats:** txt, markdown, pdf, docx, html, epub

---

### troubleshooting

**"Failed to create llama_context" (out of memory)**
your gpu ran out of vram. try:
1. close other gpu apps
2. use a smaller model
3. reduce `DEFAULT_CONTEXT_SIZE` in `chatbot/config.py`

**"CUDA not available"**
hermit is using cpu instead of gpu:
1. check nvidia drivers: `nvidia-smi`
2. re-run `./setup.sh` to rebuild torch with cuda

**"Dependencies missing"**
```bash
./setup.sh  # re-run to fix broken packages
```

---

### roadmap
- [ ] openai-compatible endpoint support (so you can use your own llama-server/ollama/whatever).
- [ ] support for more document formats in forge.
- [ ] adaptive rag (skipping joints for simple questions).

### license
licensed under **agpl v3**. see [LICENSE](LICENSE) for the full text.

---
made with <3 for the offline-first community.
