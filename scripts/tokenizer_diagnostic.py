# -*- coding: utf-8 -*-
"""
Tokenizer diagnostic for first-token surprisal analysis.

Purpose:
  Check whether the first model token of each BA/SVO target continuation
  corresponds to a linguistically meaningful first Chinese character.

Recommended use:
  Run this as a separate Colab script/notebook. It loads tokenizers only;
  it does NOT load LLM weights and does NOT compute surprisal.

Input files expected in DATA_DIR:
  - ba_svo_llm_surprisal_800_tests_Ba_context.csv
  - ba_svo_llm_surprisal_800_tests_SVO_context.csv

Outputs:
  - tokenizer_first_token_diagnostic_all.csv
  - tokenizer_first_token_summary_by_model.csv
  - tokenizer_first_token_problem_cases.csv
  - tokenizer_first_token_sample_cases.csv
"""

# If running in a fresh Colab cell, uncomment these two lines:
# !pip -q install -U transformers sentencepiece tiktoken pandas tqdm huggingface_hub

import os
import re
import json
import math
from getpass import getpass
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# ============================================================
# 0. Paths and optional Hugging Face login
# ============================================================

DATA_DIR = "/content"  # change if your CSVs are stored elsewhere
OUTPUT_DIR = "/content/tokenizer_diagnostic_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BA_CSV = os.path.join(DATA_DIR, "ba_svo_llm_surprisal_800_tests_Ba_context.csv")
SVO_CSV = os.path.join(DATA_DIR, "ba_svo_llm_surprisal_800_tests_SVO_context.csv")

# Llama models require Hugging Face access. If you already logged in, leave this False.
DO_HF_LOGIN = False
if DO_HF_LOGIN:
    from huggingface_hub import login
    hf_token = getpass("Paste your Hugging Face token: ")
    login(token=hf_token, add_to_git_credential=False)

# Optional cache directory. You can point this to Google Drive if desired.
HF_HOME = os.environ.get("HF_HOME", "/content/hf_cache")
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# ============================================================
# 1. Model list: the 21 autoregressive LLMs used in the paper
# ============================================================

MODEL_SPECS = [
    # Qwen2.5 Base
    {"alias": "Qwen2.5-0.5B-Base", "model_id": "Qwen/Qwen2.5-0.5B"},
    {"alias": "Qwen2.5-1.5B-Base", "model_id": "Qwen/Qwen2.5-1.5B"},
    {"alias": "Qwen2.5-3B-Base", "model_id": "Qwen/Qwen2.5-3B"},
    {"alias": "Qwen2.5-7B-Base", "model_id": "Qwen/Qwen2.5-7B"},
    {"alias": "Qwen2.5-14B-Base", "model_id": "Qwen/Qwen2.5-14B"},

    # Qwen3 Base
    {"alias": "Qwen3-0.6B-Base", "model_id": "Qwen/Qwen3-0.6B-Base"},
    {"alias": "Qwen3-1.7B-Base", "model_id": "Qwen/Qwen3-1.7B-Base"},
    {"alias": "Qwen3-4B-Base", "model_id": "Qwen/Qwen3-4B-Base"},
    {"alias": "Qwen3-8B-Base", "model_id": "Qwen/Qwen3-8B-Base"},
    {"alias": "Qwen3-14B-Base", "model_id": "Qwen/Qwen3-14B-Base"},

    # Yi / GLM / InternLM
    {"alias": "Yi-1.5-6B-Base", "model_id": "01-ai/Yi-1.5-6B"},
    {"alias": "Yi-1.5-9B-Base", "model_id": "01-ai/Yi-1.5-9B"},
    {"alias": "GLM-4-9B-Base", "model_id": "zai-org/glm-4-9b-hf"},
    {"alias": "InternLM2.5-7B-Base", "model_id": "internlm/internlm2_5-7b"},

    # English-centric / multilingual comparison models
    {"alias": "Pythia-410M-Base", "model_id": "EleutherAI/pythia-410m"},
    {"alias": "Pythia-1.4B-Base", "model_id": "EleutherAI/pythia-1.4b"},
    {"alias": "Pythia-2.8B-Base", "model_id": "EleutherAI/pythia-2.8b"},
    {"alias": "Mistral-7B-v0.3-Base", "model_id": "mistralai/Mistral-7B-v0.3"},
    {"alias": "Llama-3-8B-Base", "model_id": "meta-llama/Meta-Llama-3-8B"},
    {"alias": "Llama-2-7B-Base", "model_id": "meta-llama/Llama-2-7b-hf"},
    {"alias": "Llama-3.1-8B-Instruct", "model_id": "meta-llama/Llama-3.1-8B-Instruct"},
]

# For quick testing, uncomment:
# MODEL_SPECS = MODEL_SPECS[:3]

# ============================================================
# 2. Data loading
# ============================================================

def load_contextual_rows() -> pd.DataFrame:
    df_ba = pd.read_csv(BA_CSV)
    df_svo = pd.read_csv(SVO_CSV)
    df_ba["diagnostic_source_file"] = "BA_context"
    df_svo["diagnostic_source_file"] = "SOV_context"
    df = pd.concat([df_ba, df_svo], ignore_index=True)

    required_cols = [
        "global_test_id", "pair_id", "source_context_preference",
        "construction_candidate", "preference_in_source_context",
        "prompt_for_surprisal", "target_continuation", "full_test_sentence"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["target_continuation"] = df["target_continuation"].fillna("").astype(str)
    df["prompt_for_surprisal"] = df["prompt_for_surprisal"].fillna("").astype(str)
    return df

# ============================================================
# 3. Token diagnostic helpers
# ============================================================

def safe_decode_one(tokenizer, token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except TypeError:
        # Some tokenizers may not accept clean_up_tokenization_spaces.
        return tokenizer.decode([token_id])
    except Exception as e:
        return f"<DECODE_ERROR:{type(e).__name__}>"


def safe_convert_one(tokenizer, token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return str(tokenizer.convert_ids_to_tokens(token_id))
    except Exception as e:
        return f"<CONVERT_ERROR:{type(e).__name__}>"


def safe_convert_many(tokenizer, ids: List[int]) -> str:
    try:
        toks = tokenizer.convert_ids_to_tokens(ids)
        return json.dumps([str(t) for t in toks], ensure_ascii=False)
    except Exception:
        return json.dumps([], ensure_ascii=False)


def expected_first_char(continuation: str) -> str:
    cont = str(continuation).strip()
    return cont[0] if cont else ""


def classify_first_token(decoded_first: str, token_piece: str, expected_char: str) -> str:
    if not expected_char:
        return "EMPTY_CONTINUATION"
    if decoded_first == "":
        return "EMPTY_DECODED_FIRST_TOKEN"

    decoded_stripped = decoded_first.strip()

    # Best case: first model token decodes to something beginning with the expected first char.
    # This includes tokens like "把" and multi-character tokens like "把它".
    if decoded_stripped.startswith(expected_char):
        return "OK_STARTS_WITH_EXPECTED_CHAR"

    # Weaker case: expected char appears inside decoded token, but not at the beginning.
    if expected_char in decoded_stripped:
        return "WARN_CONTAINS_BUT_NOT_STARTS_WITH_EXPECTED_CHAR"

    # Common byte/fragment indicators.
    if "<0x" in token_piece or "�" in decoded_first or "�" in token_piece:
        return "WARN_BYTE_OR_FRAGMENT_TOKEN"

    # SentencePiece sometimes uses the underline symbol for whitespace; not necessarily bad,
    # but if the expected Chinese char is absent, it is not a stable linguistic first-token measure.
    if token_piece.startswith("▁") and expected_char not in decoded_stripped:
        return "WARN_SENTENCEPIECE_PREFIX_NOT_EXPECTED_CHAR"

    return "WARN_FIRST_TOKEN_NOT_EXPECTED_CHAR"


def diagnose_one_tokenizer(tokenizer, model_alias: str, model_id: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        continuation = str(row["target_continuation"]).strip()
        ids = tokenizer.encode(continuation, add_special_tokens=False)
        first_id = ids[0] if len(ids) > 0 else None
        first_decoded = safe_decode_one(tokenizer, first_id)
        first_piece = safe_convert_one(tokenizer, first_id)
        exp_char = expected_first_char(continuation)
        status = classify_first_token(first_decoded, first_piece, exp_char)

        first_5_ids = ids[:5]
        try:
            first_5_decoded = tokenizer.decode(first_5_ids, clean_up_tokenization_spaces=False)
        except Exception:
            first_5_decoded = ""

        rows.append({
            "model": model_alias,
            "model_id": model_id,
            "global_test_id": row.get("global_test_id", ""),
            "pair_id": row.get("pair_id", ""),
            "source_context_preference": row.get("source_context_preference", ""),
            "construction_candidate": row.get("construction_candidate", ""),
            "preference_in_source_context": row.get("preference_in_source_context", ""),
            "prompt_for_surprisal": row.get("prompt_for_surprisal", ""),
            "target_continuation": continuation,
            "full_test_sentence": row.get("full_test_sentence", ""),
            "expected_first_char": exp_char,
            "continuation_token_count": len(ids),
            "first_token_id": first_id,
            "first_token_piece": first_piece,
            "first_token_decoded": first_decoded,
            "first_token_decoded_repr": repr(first_decoded),
            "first_token_exact_expected_char": first_decoded.strip() == exp_char,
            "first_token_startswith_expected_char": first_decoded.strip().startswith(exp_char) if exp_char else False,
            "first_token_contains_expected_char": exp_char in first_decoded if exp_char else False,
            "diagnostic_status": status,
            "first_5_token_ids": json.dumps(first_5_ids, ensure_ascii=False),
            "first_5_token_pieces": safe_convert_many(tokenizer, first_5_ids),
            "first_5_decoded": first_5_decoded,
        })
    return pd.DataFrame(rows)

# ============================================================
# 4. Run diagnostic
# ============================================================

df_context = load_contextual_rows()
print(f"Loaded {len(df_context)} contextual target continuations.")
print(df_context[["source_context_preference", "construction_candidate"]].value_counts().sort_index())

all_results = []
failed_models = []

for spec in tqdm(MODEL_SPECS, desc="Tokenizers"):
    alias = spec["alias"]
    model_id = spec["model_id"]
    print(f"\n[Tokenizer] {alias} | {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=HF_HOME,
        )
        model_df = diagnose_one_tokenizer(tokenizer, alias, model_id, df_context)
        all_results.append(model_df)
    except Exception as e:
        print(f"[FAILED] {alias}: {type(e).__name__}: {e}")
        failed_models.append({"model": alias, "model_id": model_id, "error": repr(e)})

if not all_results:
    raise RuntimeError("No tokenizer diagnostics were completed.")

out = pd.concat(all_results, ignore_index=True)

# Summary by model and construction type.
def rate(x):
    return float(pd.Series(x).mean()) if len(x) else float("nan")

summary = (
    out.groupby("model")
    .agg(
        n_items=("global_test_id", "count"),
        startswith_rate=("first_token_startswith_expected_char", rate),
        contains_rate=("first_token_contains_expected_char", rate),
        exact_char_rate=("first_token_exact_expected_char", rate),
        n_warn=("diagnostic_status", lambda s: int((~s.astype(str).str.startswith("OK_")).sum())),
        warn_rate=("diagnostic_status", lambda s: float((~s.astype(str).str.startswith("OK_")).mean())),
    )
    .reset_index()
    .sort_values(["warn_rate", "model"], ascending=[False, True])
)

# More detailed summary by model × BA/SVO continuation.
summary_by_construction = (
    out.groupby(["model", "construction_candidate"])
    .agg(
        n_items=("global_test_id", "count"),
        startswith_rate=("first_token_startswith_expected_char", rate),
        contains_rate=("first_token_contains_expected_char", rate),
        exact_char_rate=("first_token_exact_expected_char", rate),
        warn_rate=("diagnostic_status", lambda s: float((~s.astype(str).str.startswith("OK_")).mean())),
    )
    .reset_index()
)

# Merge detailed construction rates into summary for easier reading.
ba_summary = summary_by_construction[summary_by_construction["construction_candidate"].eq("BA")][["model", "startswith_rate", "warn_rate"]].rename(columns={"startswith_rate": "ba_startswith_rate", "warn_rate": "ba_warn_rate"})
svo_summary = summary_by_construction[summary_by_construction["construction_candidate"].eq("SVO")][["model", "startswith_rate", "warn_rate"]].rename(columns={"startswith_rate": "svo_startswith_rate", "warn_rate": "svo_warn_rate"})
summary = summary.merge(ba_summary, on="model", how="left").merge(svo_summary, on="model", how="left")

problem_cases = out[~out["diagnostic_status"].astype(str).str.startswith("OK_")].copy()
sample_cases = out.groupby(["model", "construction_candidate"], group_keys=False).head(5).copy()

all_path = os.path.join(OUTPUT_DIR, "tokenizer_first_token_diagnostic_all.csv")
summary_path = os.path.join(OUTPUT_DIR, "tokenizer_first_token_summary_by_model.csv")
construction_summary_path = os.path.join(OUTPUT_DIR, "tokenizer_first_token_summary_by_model_and_construction.csv")
problem_path = os.path.join(OUTPUT_DIR, "tokenizer_first_token_problem_cases.csv")
sample_path = os.path.join(OUTPUT_DIR, "tokenizer_first_token_sample_cases.csv")
failed_path = os.path.join(OUTPUT_DIR, "tokenizer_failed_models.csv")

out.to_csv(all_path, index=False, encoding="utf-8-sig")
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
summary_by_construction.to_csv(construction_summary_path, index=False, encoding="utf-8-sig")
problem_cases.to_csv(problem_path, index=False, encoding="utf-8-sig")
sample_cases.to_csv(sample_path, index=False, encoding="utf-8-sig")
pd.DataFrame(failed_models).to_csv(failed_path, index=False, encoding="utf-8-sig")

print("\nSaved outputs:")
for p in [all_path, summary_path, construction_summary_path, problem_path, sample_path, failed_path]:
    print(" -", p)

print("\nSummary by model:")
print(summary.to_string(index=False))

print("\nInterpretation:")
print("- startswith_rate close to 1.0: first model token usually begins with the expected first Chinese character.")
print("- warn_rate high: first-token surprisal may be linguistically unstable for that tokenizer.")
print("- This diagnostic does not change the computed surprisal; it only checks whether the first-token metric is interpretable.")
