#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 correction sanity check for BA/SVO minimal-pair dataset.

Purpose
-------
This script runs a T5-based Chinese correction model over the BA/SVO test
sentences and summarizes whether the model proposes surface-level edits.

Important interpretation
------------------------
This is NOT a proof that the sentences are syntactically well-formed.
It is an auxiliary surface-correction sanity check. Human review and the
controlled minimal-pair design remain the primary evidence for syntactic
well-formedness. The T5 results can be reported as: an independent
edit-based correction model also tends not to rewrite the BA/SVO contrast
into the paired preferred construction.

Default model
-------------
shibing624/mengzi-t5-base-chinese-correction
A T5-based Chinese text correction model. You can change MODEL_NAME below.

How to run in Colab
-------------------
1. Upload this script and your CSV files to /content.
2. Run:
   %run t5_correction_sanity_check_colab.py

Outputs are written to:
   /content/t5_correction_sanity_results/
"""

# ==========================
# User configuration
# ==========================

MODEL_NAME = "shibing624/mengzi-t5-base-chinese-correction"

# Data loading:
# Prefer the all-in-one CSV if present. Otherwise the script will concatenate
# the four split CSV files.
DATA_DIR = "/content"
ALL_CSV = "ba_svo_llm_surprisal_800_tests_all.csv"
CONTEXT_CSVS = [
    "ba_svo_llm_surprisal_800_tests_Ba_context.csv",
    "ba_svo_llm_surprisal_800_tests_SVO_context.csv",
]
NO_CONTEXT_CSVS = [
    "ba_svo_llm_surprisal_800_tests_Ba_nocontext.csv",
    "ba_svo_llm_surprisal_800_tests_SVO_nocontext.csv",
]

OUTPUT_DIR = "/content/t5_correction_sanity_results"

# Run controls
RUN_ALL_ROWS = True       # True: run all available rows. False: run MAX_ROWS only.
MAX_ROWS = 30             # used only if RUN_ALL_ROWS=False
BATCH_SIZE = 16
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 128
NUM_BEAMS = 4

# Backend:
# "pycorrector" is recommended for this model. If pycorrector import/load fails,
# the script falls back to Hugging Face Transformers seq2seq generation.
BACKEND = "auto"  # "auto", "pycorrector", or "transformers"

# Whether to install missing Python packages automatically in Colab.
AUTO_INSTALL = True


# ==========================
# Imports and setup
# ==========================

import os
import re
import sys
import json
import math
import time
import shutil
import difflib
import random
import unicodedata
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

def _ensure_package(import_name: str, pip_name: Optional[str] = None):
    import importlib.util
    if importlib.util.find_spec(import_name) is None:
        if not AUTO_INSTALL:
            raise ImportError(f"Package {import_name} is missing. Install {pip_name or import_name}.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])

_ensure_package("pandas")
_ensure_package("tqdm")
_ensure_package("transformers")
_ensure_package("sentencepiece")
_ensure_package("torch")
if BACKEND in ("auto", "pycorrector"):
    try:
        _ensure_package("pycorrector")
    except Exception as e:
        print(f"[WARN] Could not install pycorrector automatically: {e}")

import pandas as pd
from tqdm.auto import tqdm

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ==========================
# Text normalization utilities
# ==========================

TERMINAL_PUNCT = "。！？!?；;，,、.．：:"
ALL_SPACE_RE = re.compile(r"\s+")

def normalize_for_comparison(text: Any, remove_terminal_punct: bool = True) -> str:
    """Normalize text for substantive correction comparison.

    Removes whitespace, normalizes unicode width, and optionally removes
    terminal punctuation. This prevents punctuation-only changes from being
    counted as substantive corrections.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = ALL_SPACE_RE.sub("", s)
    if remove_terminal_punct:
        s = s.strip(TERMINAL_PUNCT)
    return s

def simple_levenshtein(a: str, b: str) -> int:
    """Small Levenshtein implementation to avoid external dependencies."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,      # insertion
                prev[j - 1] + (ca != cb)  # substitution
            ))
        prev = cur
    return prev[-1]

def edit_summary(src: str, tgt: str, max_ops: int = 8) -> str:
    """Human-readable char-level edit summary."""
    s = normalize_for_comparison(src, remove_terminal_punct=False)
    t = normalize_for_comparison(tgt, remove_terminal_punct=False)
    ops = []
    sm = difflib.SequenceMatcher(a=s, b=t)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops.append(f"{tag}:{s[i1:i2]}->{t[j1:j2]}@{i1}:{i2}")
        if len(ops) >= max_ops:
            ops.append("...")
            break
    return " | ".join(ops)


# ==========================
# Data loading
# ==========================

def find_file(data_dir: str, fname: str) -> Optional[str]:
    p = Path(data_dir) / fname
    return str(p) if p.exists() else None

def load_dataset() -> pd.DataFrame:
    all_path = find_file(DATA_DIR, ALL_CSV)
    if all_path:
        print(f"[INFO] Loading all-in-one CSV: {all_path}")
        df = pd.read_csv(all_path)
    else:
        print("[INFO] All-in-one CSV not found; concatenating split CSV files.")
        frames = []
        for fname in CONTEXT_CSVS + NO_CONTEXT_CSVS:
            p = find_file(DATA_DIR, fname)
            if not p:
                raise FileNotFoundError(f"Could not find {fname} in {DATA_DIR}")
            frames.append(pd.read_csv(p))
        df = pd.concat(frames, ignore_index=True)

    required = [
        "global_test_id", "context_status", "source_context_preference",
        "subcondition_label", "pair_id", "construction_candidate",
        "preference_in_source_context", "prompt_for_surprisal",
        "target_continuation", "full_test_sentence"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}\nColumns: {df.columns.tolist()}")

    # Deduplicate if needed.
    df = df.drop_duplicates(subset=["global_test_id"]).copy()
    df["row_id"] = range(len(df))
    df["sentence_for_t5"] = df["full_test_sentence"].astype(str)

    # Main labels
    df["is_contextualized"] = df["context_status"].astype(str).str.contains("有语境", na=False)
    df["is_contextless"] = df["context_status"].astype(str).str.contains("无语境", na=False)
    df["gold_is_nonpreferred"] = df["preference_in_source_context"].astype(str).eq("non-preferred")
    df["gold_is_preferred"] = df["preference_in_source_context"].astype(str).eq("preferred")

    # Optional quick run
    if not RUN_ALL_ROWS:
        df = df.head(MAX_ROWS).copy()
        print(f"[INFO] Quick mode: running first {len(df)} rows only.")
    else:
        print(f"[INFO] Full mode: running {len(df)} rows.")

    return df

df = load_dataset()
print(df[["global_test_id", "context_status", "source_context_preference",
          "construction_candidate", "preference_in_source_context",
          "full_test_sentence"]].head(6).to_string(index=False))


# ==========================
# T5 correction backends
# ==========================

def load_pycorrector_backend(model_name: str):
    from pycorrector.t5.t5_corrector import T5Corrector
    corrector = T5Corrector(model_name)
    return corrector

def pycorrector_batch_correct(corrector, texts: List[str]) -> List[Dict[str, Any]]:
    """Run pycorrector T5 batch correction and parse outputs robustly."""
    raw = corrector.batch_t5_correct(texts)
    parsed = []
    for x in raw:
        corrected, details = None, None
        if isinstance(x, tuple):
            corrected = x[0]
            details = x[1] if len(x) > 1 else None
        elif isinstance(x, list) and len(x) >= 1:
            # Some versions may return [corrected, details]
            corrected = x[0]
            details = x[1] if len(x) > 1 else None
        else:
            corrected = str(x)
            details = None
        parsed.append({
            "t5_corrected": str(corrected),
            "t5_details": json.dumps(details, ensure_ascii=False) if details is not None else "",
            "t5_num_reported_edits": len(details) if isinstance(details, list) else None,
        })
    return parsed

def load_transformers_backend(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

def transformers_batch_correct(tok_model_device, texts: List[str]) -> List[Dict[str, Any]]:
    import torch
    tokenizer, model, device = tok_model_device
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False
        )
    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    return [{
        "t5_corrected": str(s),
        "t5_details": "",
        "t5_num_reported_edits": None,
    } for s in decoded]

def run_t5_correction(texts: List[str]) -> Tuple[List[Dict[str, Any]], str]:
    backend_used = None

    if BACKEND in ("auto", "pycorrector"):
        try:
            print(f"[INFO] Loading pycorrector T5 backend: {MODEL_NAME}")
            corrector = load_pycorrector_backend(MODEL_NAME)
            results = []
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="T5 correction (pycorrector)"):
                batch = texts[i:i+BATCH_SIZE]
                results.extend(pycorrector_batch_correct(corrector, batch))
            backend_used = "pycorrector"
            return results, backend_used
        except Exception as e:
            if BACKEND == "pycorrector":
                raise
            print(f"[WARN] pycorrector backend failed; falling back to Transformers. Error: {repr(e)}")

    print(f"[INFO] Loading Transformers seq2seq backend: {MODEL_NAME}")
    tm = load_transformers_backend(MODEL_NAME)
    results = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="T5 correction (transformers)"):
        batch = texts[i:i+BATCH_SIZE]
        results.extend(transformers_batch_correct(tm, batch))
    backend_used = "transformers"
    return results, backend_used


texts = df["sentence_for_t5"].tolist()
t0 = time.time()
corrections, backend_used = run_t5_correction(texts)
elapsed = time.time() - t0
print(f"[INFO] Finished T5 correction using backend={backend_used}; elapsed={elapsed:.1f}s")

corr_df = pd.DataFrame(corrections)
df = pd.concat([df.reset_index(drop=True), corr_df.reset_index(drop=True)], axis=1)


# ==========================
# Compute correction flags
# ==========================

df["input_norm"] = df["sentence_for_t5"].map(normalize_for_comparison)
df["t5_corrected_norm"] = df["t5_corrected"].map(normalize_for_comparison)
df["t5_raw_changed"] = df["sentence_for_t5"].astype(str).str.strip() != df["t5_corrected"].astype(str).str.strip()
df["t5_substantive_changed"] = df["input_norm"] != df["t5_corrected_norm"]
df["t5_accepted_no_substantive_edit"] = ~df["t5_substantive_changed"]
df["t5_edit_distance_norm"] = [
    simple_levenshtein(a, b) for a, b in zip(df["input_norm"], df["t5_corrected_norm"])
]
df["t5_edit_summary"] = [
    edit_summary(a, b) for a, b in zip(df["sentence_for_t5"], df["t5_corrected"])
]

# Pair-level preferred sentence map for contextualized rows.
context_df_all = df[df["is_contextualized"]].copy()
preferred_map = (
    context_df_all[context_df_all["gold_is_preferred"]]
    .set_index("pair_id")["full_test_sentence"]
    .to_dict()
)

df["paired_preferred_sentence"] = df["pair_id"].map(preferred_map)
df["paired_preferred_norm"] = df["paired_preferred_sentence"].map(normalize_for_comparison)
df["t5_corrected_to_paired_preferred"] = (
    df["is_contextualized"]
    & df["gold_is_nonpreferred"]
    & (df["t5_corrected_norm"] == df["paired_preferred_norm"])
)


# ==========================
# Summary functions
# ==========================

def summarize_subset(sub: pd.DataFrame, scope: str) -> Dict[str, Any]:
    n = len(sub)
    if n == 0:
        return {"scope": scope, "N": 0}
    nonpref = sub[sub["gold_is_nonpreferred"]]
    pref = sub[sub["gold_is_preferred"]]

    out = {
        "scope": scope,
        "N": n,
        "substantive_changed_N": int(sub["t5_substantive_changed"].sum()),
        "substantive_changed_rate": float(sub["t5_substantive_changed"].mean()),
        "accepted_no_substantive_edit_N": int(sub["t5_accepted_no_substantive_edit"].sum()),
        "accepted_no_substantive_edit_rate": float(sub["t5_accepted_no_substantive_edit"].mean()),
        "mean_edit_distance_norm": float(sub["t5_edit_distance_norm"].mean()),
        "median_edit_distance_norm": float(sub["t5_edit_distance_norm"].median()),
        "nonpreferred_N": len(nonpref),
        "nonpreferred_flagged_N": int(nonpref["t5_substantive_changed"].sum()) if len(nonpref) else 0,
        "nonpreferred_flagged_rate": float(nonpref["t5_substantive_changed"].mean()) if len(nonpref) else None,
        "nonpreferred_miss_rate": float((~nonpref["t5_substantive_changed"]).mean()) if len(nonpref) else None,
        "preferred_N": len(pref),
        "preferred_accepted_N": int((~pref["t5_substantive_changed"]).sum()) if len(pref) else 0,
        "preferred_acceptance_rate": float((~pref["t5_substantive_changed"]).mean()) if len(pref) else None,
        "preferred_false_positive_rate": float(pref["t5_substantive_changed"].mean()) if len(pref) else None,
        "strict_construction_level_correction_N": int(sub["t5_corrected_to_paired_preferred"].sum()),
        "strict_construction_level_correction_rate_over_nonpreferred": (
            float(sub["t5_corrected_to_paired_preferred"].sum() / len(nonpref)) if len(nonpref) else None
        ),
    }
    return out

summary_rows = []
summary_rows.append(summarize_subset(df, "all_800_scored_conditions_or_loaded_rows"))
summary_rows.append(summarize_subset(df[df["is_contextualized"]], "contextualized_400_target_sentences"))
summary_rows.append(summarize_subset(df[df["is_contextless"]], "contextless_400_target_sentences"))

# By source context preference and candidate
for keys, name in [
    (["context_status"], "by_context_status"),
    (["context_status", "source_context_preference"], "by_context_status_and_preference"),
    (["context_status", "source_context_preference", "construction_candidate"], "by_context_and_candidate"),
    (["context_status", "source_context_preference", "preference_in_source_context"], "by_context_and_gold_preference"),
]:
    for group_vals, sub in df.groupby(keys, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        scope = name + "__" + "__".join([f"{k}={v}" for k, v in zip(keys, group_vals)])
        summary_rows.append(summarize_subset(sub, scope))

summary_df = pd.DataFrame(summary_rows)

# Contextual pair-level summary.
pair_rows = []
ctx = df[df["is_contextualized"]].copy()
for pair_id, g in ctx.groupby("pair_id"):
    preferred = g[g["gold_is_preferred"]]
    nonpreferred = g[g["gold_is_nonpreferred"]]
    if len(preferred) != 1 or len(nonpreferred) != 1:
        continue
    pref_row = preferred.iloc[0]
    nonpref_row = nonpreferred.iloc[0]
    pair_rows.append({
        "pair_id": pair_id,
        "source_context_preference": pref_row["source_context_preference"],
        "subcondition_label": pref_row["subcondition_label"],
        "preferred_construction": pref_row["construction_candidate"],
        "nonpreferred_construction": nonpref_row["construction_candidate"],
        "preferred_sentence": pref_row["full_test_sentence"],
        "nonpreferred_sentence": nonpref_row["full_test_sentence"],
        "preferred_corrected": pref_row["t5_corrected"],
        "nonpreferred_corrected": nonpref_row["t5_corrected"],
        "preferred_accepted": bool(not pref_row["t5_substantive_changed"]),
        "nonpreferred_flagged": bool(nonpref_row["t5_substantive_changed"]),
        "pair_edit_accept_correct": bool((not pref_row["t5_substantive_changed"]) and nonpref_row["t5_substantive_changed"]),
        "nonpreferred_corrected_to_paired_preferred": bool(nonpref_row["t5_corrected_to_paired_preferred"]),
        "preferred_edit_summary": pref_row["t5_edit_summary"],
        "nonpreferred_edit_summary": nonpref_row["t5_edit_summary"],
    })
pair_df = pd.DataFrame(pair_rows)

def summarize_pairs(pair_df: pd.DataFrame, scope: str) -> Dict[str, Any]:
    n = len(pair_df)
    if n == 0:
        return {"scope": scope, "N_pairs": 0}
    return {
        "scope": scope,
        "N_pairs": n,
        "preferred_acceptance_rate": float(pair_df["preferred_accepted"].mean()),
        "nonpreferred_detection_rate": float(pair_df["nonpreferred_flagged"].mean()),
        "nonpreferred_miss_rate": float((~pair_df["nonpreferred_flagged"]).mean()),
        "pair_edit_accept_accuracy": float(pair_df["pair_edit_accept_correct"].mean()),
        "strict_construction_level_correction_rate": float(pair_df["nonpreferred_corrected_to_paired_preferred"].mean()),
        "strict_construction_level_correction_N": int(pair_df["nonpreferred_corrected_to_paired_preferred"].sum()),
    }

pair_summary_rows = []
pair_summary_rows.append(summarize_pairs(pair_df, "all_contextualized_pairs"))
for pref, sub in pair_df.groupby("source_context_preference"):
    pair_summary_rows.append(summarize_pairs(sub, f"{pref}_pairs"))
for subcond, sub in pair_df.groupby("subcondition_label"):
    pair_summary_rows.append(summarize_pairs(sub, f"subcondition={subcond}"))
pair_summary_df = pd.DataFrame(pair_summary_rows)


# Appendix-friendly compact table
appendix_summary = pd.DataFrame([
    summarize_subset(df[df["is_contextualized"]], "T5 correction sanity check: contextualized sentences"),
    summarize_subset(df[df["is_contextless"]], "T5 correction sanity check: contextless sentences"),
    summarize_subset(df, "T5 correction sanity check: all loaded sentences"),
])
keep_cols = [
    "scope", "N", "accepted_no_substantive_edit_rate", "substantive_changed_rate",
    "nonpreferred_miss_rate", "preferred_acceptance_rate",
    "strict_construction_level_correction_rate_over_nonpreferred"
]
appendix_summary = appendix_summary[keep_cols]


# Changed cases for manual inspection
changed_cases = df[df["t5_substantive_changed"]].copy()
changed_cases = changed_cases[[
    "global_test_id", "context_status", "source_context_preference",
    "subcondition_label", "pair_id", "construction_candidate",
    "preference_in_source_context", "sentence_for_t5", "t5_corrected",
    "t5_edit_distance_norm", "t5_edit_summary", "t5_details",
    "t5_corrected_to_paired_preferred"
]]

# Save outputs
output_files = {
    "sentence_results": Path(OUTPUT_DIR) / "t5_all_sentence_results.csv",
    "summary": Path(OUTPUT_DIR) / "t5_summary_metrics.csv",
    "pair_results": Path(OUTPUT_DIR) / "t5_contextual_pair_level_results.csv",
    "pair_summary": Path(OUTPUT_DIR) / "t5_contextual_pair_summary_metrics.csv",
    "changed_cases": Path(OUTPUT_DIR) / "t5_changed_cases_for_manual_review.csv",
    "appendix_summary": Path(OUTPUT_DIR) / "t5_appendix_summary_table.csv",
    "run_config": Path(OUTPUT_DIR) / "t5_run_config.json",
}

df.to_csv(output_files["sentence_results"], index=False, encoding="utf-8-sig")
summary_df.to_csv(output_files["summary"], index=False, encoding="utf-8-sig")
pair_df.to_csv(output_files["pair_results"], index=False, encoding="utf-8-sig")
pair_summary_df.to_csv(output_files["pair_summary"], index=False, encoding="utf-8-sig")
changed_cases.to_csv(output_files["changed_cases"], index=False, encoding="utf-8-sig")
appendix_summary.to_csv(output_files["appendix_summary"], index=False, encoding="utf-8-sig")

config = {
    "MODEL_NAME": MODEL_NAME,
    "BACKEND_USED": backend_used,
    "DATA_DIR": DATA_DIR,
    "OUTPUT_DIR": OUTPUT_DIR,
    "RUN_ALL_ROWS": RUN_ALL_ROWS,
    "MAX_ROWS": MAX_ROWS,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_INPUT_LENGTH": MAX_INPUT_LENGTH,
    "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "NUM_BEAMS": NUM_BEAMS,
    "elapsed_seconds": elapsed,
    "n_rows": len(df),
}
with open(output_files["run_config"], "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

# Print summaries
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

print("\n================ T5 sentence-level summary ================")
print(appendix_summary.to_string(index=False))

print("\n================ T5 contextual pair-level summary ================")
print(pair_summary_df.to_string(index=False))

print("\n================ Example changed cases ================")
if len(changed_cases):
    print(changed_cases.head(20)[[
        "global_test_id", "context_status", "source_context_preference",
        "construction_candidate", "preference_in_source_context",
        "sentence_for_t5", "t5_corrected", "t5_edit_summary",
        "t5_corrected_to_paired_preferred"
    ]].to_string(index=False))
else:
    print("No substantive changed cases.")

print("\n================ Saved files ================")
for k, p in output_files.items():
    print(f"{k}: {p}")

print("\nInterpretation note:")
print("- This is an auxiliary edit-based sanity check, not a proof of syntactic well-formedness.")
print("- Human review and controlled item design remain the primary syntactic/pragmatic validation.")
print("- A low construction-level correction rate means T5 did not rewrite non-preferred BA/SVO continuations into the paired preferred alternatives.")
print("- Review t5_changed_cases_for_manual_review.csv before reporting any claim in the paper.")
