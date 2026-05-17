# To "Ba" or Not to "Ba"? 
**Disentangling Pragmatic Constraints and Local Collocations in L2 Chinese Evaluation via LLM Surprisal**

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026_Submission-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Anonymous Repository:** This repository contains the official dataset, evaluation scripts, and raw surprisal logs for our EMNLP 2026 double-blind submission. 

## 📖 Overview

The Chinese *ba* (把) construction poses significant challenges for second language (L2) learners, frequently resulting in implicit pragmatic errors such as avoidance or overgeneralization. Traditional Chinese Grammatical Error Correction (CGEC) systems suffer from "pragmatic blindness" and systematically fail to detect these contextually infelicitous but syntactically well-formed sentences.

This repository open-sources a novel paradigm that leverages the autoregressive surprisal of Large Language Models (LLMs) to evaluate L2 pragmatic naturalness. We comprehensively evaluate 23 LLMs and a traditional baseline (MacBERT) across a tightly controlled dataset.

### Key Findings:
1. **H1 (Baseline Failure):** Explicit CGEC baselines (MacBERT) exhibit a 100% miss rate on pragmatic deviations, while LLM surprisal achieves up to 96.0% accuracy.
2. **H2 (Context-Driven Coercion):** Pragmatic contexts strongly facilitate target structures, drastically reducing surprisal and driving robust structural preference reversals.
3. **H3 (Constructional Gestalt):** Sentence-level surprisal significantly outperforms Word-level (first-token) metrics, proving that the *ba* construction operates as a multi-token gestalt rather than an instantaneous syntactic divergence.

---

## 📂 Repository Structure

```text
.
├── data/                       # The 800-sentence controlled evaluation dataset
│   ├── BA_With_Context_200.csv
│   ├── SVO_With_Context_200.csv
│   ├── BA_Without_Context_200.csv
│   └── SVO_Without_Context_200.csv
│
├── scripts/                    # Source code for reproducibility
│   ├── 1_macbert_baseline.py   # CGEC baseline evaluation script
│   ├── 2_llm_surprisal.py      # Autoregressive surprisal extraction for 23 LLMs
│   └── 3_statistical_analysis.py # Accuracy calculation, T-tests, and plotting
│
├── results/                    # Generated outputs and visualizations
│   ├── raw_surprisal_logs/     # Raw CSV outputs from all 23 evaluated LLMs
│   └── figures/                # High-resolution charts (Boxplots, Scatter, Dumbbell)
│
├── requirements.txt            # Python dependencies
└── README.md
