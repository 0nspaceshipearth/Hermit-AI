# hermit

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Accelerated-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

offline ai chatbot for wikipedia and zim archives. no cloud, no api keys, no tracking.

---

<img width="895" height="697" alt="hermit screenshot" src="https://github.com/user-attachments/assets/a60de92a-38cf-42a8-bd31-ca96429d5bf5" />

---

### what hermit does differently

instead of trusting a single vector search, hermit chains multiple model calls together. each one checks the work of the previous step. the system extracts entities from your question, predicts which articles are relevant, scores them for actual relevance, filters down to the useful paragraphs, and only then generates an answer. it's slower, but the answers are grounded in real content.

for details on how the architecture works, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

### setup

```bash
git clone https://github.com/imDelivered/Hermit-AI.git
cd Hermit-AI
./setup.sh
```

grab a .zim file from [kiwix.org](https://library.kiwix.org/), drop it in the folder, run `hermit`.
