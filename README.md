# Hermit

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Accelerated-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

offline ai chatbot for wikipedia and zim archives. no cloud, no api keys, no tracking.

---
<img width="901" height="693" alt="Screenshot_20260206_143725" src="https://github.com/user-attachments/assets/fb75178e-13c7-48b1-8626-48cc2c862407" />


---

### what hermit does differently

a synergistic architecture that utilizes models of various types and sizes to structure reasoning pipelines. At an architectural level, that’s the core philosophy. But what have I actually built with it?

I call it "Hermit."

It is a 100% offline .zim data extraction and context injection tool for answering queries. All you need to know is that at the end of the chain (I call them "Joints"), the final model gets handed the best data it found in your collection of .zim files with 100% available context.

Why .zim files? Because there is a plethora of available collections containing incredibly useful information through sources like Kiwix. I personally use this in parallel with  research papers, Wikipedia, Project Gutenberg, Stack Overflow, and medical, etc... all equating to about 300gb of data for local model context injection.   

for details on how the architecture works, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

### setup

```bash
git clone https://github.com/imDelivered/Hermit-AI.git
cd Hermit-AI
./setup.sh
```

grab a .zim file from [kiwix.org](https://library.kiwix.org/), drop it in the folder, run `hermit`.
