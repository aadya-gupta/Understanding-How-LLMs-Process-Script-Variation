# Understanding-How-LLMs-Process-Script-Variation

This repository contains the datasets, evaluation scripts, and mechanistic interpretability probes necessary to reproduce the findings in our ARR submission. 

Our experiments investigate the emergence of **Script Invariance** in Large Language Models (LLMs) across three linguistic tasks: Named Entity Recognition (NER), Part-of-Speech (POS) Tagging, and Natural Language Inference (NLI).

---

## Hardware Requirements & Execution Environments

To optimize compute efficiency, our pipeline is split across two hardware environments:

1. **Task Accuracy Evaluation Scripts:** Optimized for **Apple Silicon (MPS)**. These scripts bypass specific C++ attention crashes on Mac by forcing `eager` attention execution.
2. **Hidden State Extraction & Probing:** Optimized for **CUDA (Google Colab Free Tier - T4 GPU)**. These scripts utilize `bitsandbytes` 4-bit quantization and atomic batched inference to fit entirely within the 15GB VRAM limit.

---
