# To "Ba" or Not to "Ba"? 
## LLM Surprisal as a Diagnostic of Context-Conditioned Naturalness in L2 Chinese

[![LUHME 2026](https://img.shields.io/badge/LUHME-2026_Submission-blue.svg)](https://luhme.up.pt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Anonymous Repository:** This repository contains the dataset, evaluation scripts, raw model outputs, and supplementary checks for a study on Mandarin **ba** (把) in L2 Chinese. The study asks whether autoregressive LLM surprisal can help diagnose cases where two continuations are both syntactically well formed, but only one is natural in the preceding discourse context.

## Overview

The Mandarin **ba** construction is difficult for L2 Chinese learners because the problem is not always visible as a surface grammatical error. Learners may avoid **ba** when the context favors it, or overextend **ba** to contexts where an SVO sentence is more natural.

For example:

```text
Context:
那个苹果坏了，我……
That apple went bad, I...

Preferred continuation:
把它扔了。
...threw it away.

Non-preferred continuation:
扔了它。
...threw it away.
```

Both continuations can be syntactically possible, but they differ in contextual naturalness. This makes the phenomenon difficult for edit-based correction systems, which usually decide whether to edit or accept a sentence.

We test three questions:

1. **H1: Correction baseline vs. LLM surprisal**
   Does an edit-based correction baseline flag pragmatically infelicitous BA/SVO alternatives? Does sentence-level LLM surprisal select the contextually preferred continuation?

2. **H2: Contextual facilitation**
   Does adding the licensing context reduce the surprisal of the continuation it supports?

3. **H3: Granularity of surprisal**
   Is the relevant signal better captured by sentence-level surprisal over the full continuation, or by first-token surprisal at the initial BA/SVO choice point?

---

## Key findings

* A MacBERT-based correction baseline misses **97.0%** of pragmatically infelicitous continuations and produces **0.0%** construction-level BA/SVO corrections.

* Across 21 open-weight autoregressive LLMs, sentence-level surprisal selects the contextually preferred continuation with an average accuracy of **87.93%**. The best model reaches **93.50%**.

* Context generally reduces surprisal for the licensed continuation. We report this as:

  ```text
  ΔS = sentence-level surprisal without context − sentence-level surprisal with context
  ```

  Positive ΔS means that the same continuation becomes more expected after the relevant context is supplied.

* Sentence-level surprisal is more reliable than first-token surprisal. The average sentence-level accuracy is **87.93%**, compared with **57.50%** for first-token surprisal.

* A tokenizer diagnostic shows that first-token surprisal is not always linguistically interpretable for Chinese. In some model families, the first model token does not correspond to a meaningful Chinese character such as 把.

---

## Repository structure

```text
.
├── data/
│   ├── ba_svo_llm_surprisal_800_tests_all.csv
│   ├── ba_svo_llm_surprisal_800_tests_BA_context.csv
│   ├── ba_svo_llm_surprisal_800_tests_SVO_context.csv
│   ├── ba_svo_llm_surprisal_800_tests_Ba_nocontext.csv
│   ├── ba_svo_llm_surprisal_800_tests_SVO_nocontext.csv
│   └── ba_svo_llm_surprisal_800_tests_Pair.csv
│
├── scripts/
│   ├── all_models_test.py
│   ├── macbert_baseline.py
│   ├── experiment_results.py
│   ├── tokenizer_diagnostic.py
│   └── t5_correction_check.py
│
├── results/
│   ├── figures/
│   ├── llm_surprisal_results/
│   ├── macbert_baseline_results/
│   └── t5_correction_sanity_results/
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Dataset

The evaluation contains **800 scored continuation conditions**.

### Contextualized condition

There are **200 contextual minimal pairs**. Each pair contains:

* one preceding discourse context,
* one **ba** continuation,
* one SVO continuation.

This gives **400 contextualized target continuations**.

The contextualized subset is divided into:

* **BA-preferred contexts**: 100 contexts where the **ba** continuation is more natural.

  * 50 physical state-change contexts.
  * 50 spatial displacement contexts.

* **SVO-preferred contexts**: 100 contexts where the SVO continuation is more natural.

  * 50 habitual-action contexts.
  * 50 narrative or routine event-description contexts.

### Contextless condition

For H2, we also remove the preceding discourse context and score the same continuations in isolation. This gives another **400 contextless scoring conditions**.

The contextless condition is used to ask whether a continuation becomes less surprising when the relevant discourse context is supplied.

---

## Metrics

### Sentence-level mean surprisal

For a continuation (T = (w_1, ..., w_N)) conditioned on context (C), sentence-level mean surprisal is the average negative log probability over the scored continuation tokens:

```text
S_sent(T | C) = (1/N) * sum_i -ln P(w_i | C, w_<i)
```

The context is used as conditioning information, but surprisal is computed only over the target continuation tokens.

### First-token surprisal

First-token surprisal is the negative log probability of the first scored model token in the target continuation:

```text
S_first(T | C) = -ln P(w_1 | C)
```

For BA continuations, this token is intended to correspond to the **ba** marker. For SVO continuations, it is intended to correspond to the first main verb or the first token of the SVO continuation. Because tokenization differs across model families, this token is not always a linguistically meaningful Chinese character.

### Surprisal reduction

For H2, we compute:

```text
ΔS = S_sent(T | no context) − S_sent(T | context)
```

A positive value means that the same continuation becomes less surprising after the relevant context is supplied. All surprisal values are reported in nats.

---

## Models

The evaluation covers 21 open-weight autoregressive LLMs and one MacBERT-based correction baseline.

The autoregressive models include:

* Qwen2.5: 0.5B, 1.5B, 3B, 7B, 14B
* Qwen3: 0.6B, 1.7B, 4B, 8B, 14B
* Llama-2-7B-Base
* Llama-3-8B-Base
* Llama-3.1-8B-Instruct
* Mistral-7B-v0.3-Base
* InternLM2.5-7B-Base
* GLM-4-9B-Base
* Yi-1.5: 6B, 9B
* Pythia: 410M, 1.4B, 2.8B

MacBERT is used as an edit-based correction baseline. It is not used as a grammaticality validator.

---

## How to reproduce

### 1. Install dependencies

The scripts were developed for Python 3.8+.

A typical Colab setup is:

```bash
pip install -U pandas numpy scipy tqdm matplotlib seaborn
pip install -U torch transformers accelerate sentencepiece protobuf
pip install -U pycorrector
```

Some model families, especially Llama models, may require a Hugging Face access token.

### 2. Run LLM surprisal scoring

```bash
python scripts/all_models_test.py
```

This script scores BA and SVO continuations with and without context and saves raw surprisal logs.

### 3. Generate main results and figures

```bash
python scripts/results.py
```

This script computes the H1, H2, and H3 metrics and generates the figures used in the paper.

### 4. Run the MacBERT correction baseline

```bash
python scripts/macbert_baseline.py
```

This script evaluates 400 full contextualized sentences. Each input is the preceding context concatenated with one target continuation.

### 5. Run tokenizer diagnostic for H3

```bash
python scripts/tokenizer_diagnostic_colab.py
```

This script checks whether the first model token of each target continuation begins with the expected first Chinese character. It does not change the surprisal values; it only checks whether the first-token metric is linguistically interpretable.

### 6. Run optional T5 correction sanity check

```bash
python scripts/t5_correction_sanity_check_colab.py
```

This script runs a T5-based Chinese correction model as an auxiliary surface-correction sanity check. It is not used to validate grammaticality. It checks whether another edit-based model systematically rewrites BA/SVO alternatives.

---

## Output files

### Main figures

The `results/figures/` folder contains:

* H1 model-wise sentence-level accuracy ranking.
* H2 surprisal reduction plots.
* H2 preference reversal plot.
* H3 sentence-level vs. first-token scatter plot.

### Raw logs

The `results/raw_surprisal_logs/` folder contains raw surprisal scores for each evaluated LLM.

### Sanity results

The `results/t5_correction_sanity_results/` folder contains the auxiliary T5 correction sanity-check outputs.

### Baseline results

The `results/macbert_baseline_results/` folder contains sentence-level and pair-level MacBERT outputs.

### Tokenization diagnostic

The `results/tokenizer_diagnostic_results/` folder contains the first-token diagnostic used to interpret H3.
