"""
evaluation/baseline_eval.py
----------------------------
Runs all 4 approaches against every fault in the library.
Produces a side-by-side comparison table and saves results to CSV.

Approaches evaluated:
  1. Rule-Based
  2. LLM-Only  (Mistral, no context)
  3. RAG-Only  (retrieval + keyword matching, no LLM)
  4. LLM + RAG (retrieval + Mistral reasoning)

CUDA fix: llm_only and llm_rag both use retry-with-wait logic
and keep_alive=0 so the model is unloaded between calls,
preventing memory buildup that causes CUDA error 500.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, json, re
import ollama
import pandas as pd

from fault_generator import get_all_library_faults, get_random_fault
from rag_pipeline import RAG
from prompts import RECOVERY_PROMPT

MODEL      = "mistral:7b-instruct-q4_0"
CUDA_WAIT  = 8    # seconds to wait after a CUDA error before retrying
MAX_TRIES  = 3    # max LLM attempts per call before falling back to SHUTDOWN

# ─────────────────────────────────────────────
# Shared RAG instance (ingest once)
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL_PATH = os.path.join(BASE_DIR, "data", "energy_manual.txt")

print("📚 Loading RAG pipeline...")
rag = RAG(collection_name="baseline_eval")
rag.ingest(MANUAL_PATH)


# ─────────────────────────────────────────────
# SHARED LLM CALL HELPER  (CUDA-safe)
# ─────────────────────────────────────────────
def safe_llm_call(prompt: str, max_tokens: int = 150) -> str:
    """
    Wraps ollama.chat with retry logic for CUDA errors.
    Returns the raw response text, or empty string on total failure.
    """
    for attempt in range(MAX_TRIES):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": 1024, "num_predict": max_tokens, "temperature": 0.1},
                keep_alive=0   # unload model from memory after each call
            )
            return response["message"]["content"]
        except Exception as e:
            err = str(e)
            if "CUDA error" in err or "llama runner" in err:
                print(f"  ⚠️  CUDA error (attempt {attempt+1}/{MAX_TRIES}), "
                      f"waiting {CUDA_WAIT}s for model to unload...")
                time.sleep(CUDA_WAIT)
            else:
                print(f"  ⚠️  LLM error: {err}")
                break
    return ""   # total failure — caller uses SHUTDOWN fallback


def parse_action(text: str) -> str:
    """Extract and validate action from LLM JSON response."""
    try:
        text = text.replace("```json", "").replace("```", "")
        text = re.sub(r",\s*}", "}", text)    # trailing commas
        text = re.sub(r",\s*]", "]", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            action = data.get("action", "SHUTDOWN").upper()
            if action in ("RETRY", "REDIRECT", "SHUTDOWN"):
                return action
    except Exception:
        pass
    return "SHUTDOWN"


# ─────────────────────────────────────────────
# APPROACH 1: Rule-Based
# ─────────────────────────────────────────────
def rule_based(error: str) -> str:
    e = error.lower()
    if any(w in e for w in ["overload", "overloading", "thermal", "overcurrent"]):
        return "REDIRECT"
    elif any(w in e for w in ["disconnect", "loss", "partial", "sag",
                               "deviation", "imbalance"]):
        return "RETRY"
    elif any(w in e for w in ["overheating", "emergency", "breaker",
                               "shutdown", "failed", "failure"]):
        return "SHUTDOWN"
    return "SHUTDOWN"


# ─────────────────────────────────────────────
# APPROACH 2: LLM-Only
# ─────────────────────────────────────────────
LLM_ONLY_PROMPT = """You are an energy grid fault recovery system.

Error: {error}

Choose the correct recovery action.
Decision guide:
- RETRY: fault is likely transient (frequency deviation, voltage sag, temporary)
- REDIRECT: fault blocks current path but rerouting is possible (overload, disconnect)
- SHUTDOWN: fault is severe / unsafe (overheating, emergency, breaker failure)

Return ONLY valid JSON:
{{"action": "<RETRY|REDIRECT|SHUTDOWN>"}}"""


def llm_only(error: str) -> str:
    text = safe_llm_call(
        LLM_ONLY_PROMPT.format(error=error),
        max_tokens=80
    )
    return parse_action(text)


# ─────────────────────────────────────────────
# APPROACH 3: RAG-Only
# ─────────────────────────────────────────────
def rag_only(error: str) -> str:
    context = rag.query(error, n_results=6)
    c = context.lower()

    redirect_keywords = [
        "redirect", "reroute", "alternate", "parallel path",
        "redistribute", "congestion", "thermal overloading on parallel",
        "alternate route", "restore via"
    ]
    retry_keywords = [
        "auto-reclos", "transient", "restore", "reclos",
        "reclose", "temporary", "cleared", "retry"
    ]
    shutdown_keywords = [
        "isolat", "de-energi", "emergency", "severe",
        "permanent fault", "unsafe", "damage"
    ]

    redirect_score = sum(1 for kw in redirect_keywords if kw in c)
    retry_score    = sum(1 for kw in retry_keywords    if kw in c)
    shutdown_score = sum(1 for kw in shutdown_keywords if kw in c)

    scores = {
        "REDIRECT": redirect_score,
        "RETRY":    retry_score,
        "SHUTDOWN": shutdown_score
    }
    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return rule_based(error)   # fallback to rule-based when no keywords match
    return best


# ─────────────────────────────────────────────
# APPROACH 4: LLM + RAG
# ─────────────────────────────────────────────
def llm_rag(error: str) -> str:
    context = rag.query(error, n_results=6)
    prompt  = RECOVERY_PROMPT.format(error=error, context=context)
    text    = safe_llm_call(prompt, max_tokens=150)
    return parse_action(text)


# ─────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────
APPROACHES = {
    "Rule-Based": rule_based,
    "LLM-Only":   llm_only,
    "RAG-Only":   rag_only,
    "LLM+RAG":    llm_rag,
}


def evaluate_all(include_llm_faults: int = 3) -> pd.DataFrame:
    faults = get_all_library_faults()

    for i in range(include_llm_faults):
        novel = get_random_fault(fault_id=f"GEN_{i+1:03d}", llm_ratio=1.0)
        faults.append(novel)

    total_calls = len(faults) * len(APPROACHES)
    print(f"\n🔍 Evaluating {len(faults)} faults × {len(APPROACHES)} approaches "
          f"= {total_calls} total calls")
    print(f"   Estimated time: {total_calls * 20 // 60}–{total_calls * 30 // 60} minutes\n")

    rows = []

    for fault in faults:
        fid     = fault["fault_id"]
        error   = fault["error_message"]
        correct = fault["correct_action"]
        ftype   = fault["fault_type"]
        source  = fault["source"]

        print(f"  Fault {fid} [{ftype}] (source={source})")
        print(f"  Error  : {error}")
        print(f"  Correct: {correct}")

        for name, fn in APPROACHES.items():
            t0      = time.time()
            result  = fn(error)
            latency = round(time.time() - t0, 4)
            is_correct = int(result == correct)
            mark = "✅" if is_correct else "❌"

            print(f"    {name:<12} → {result:<10} {mark}  ({latency:.3f}s)")

            rows.append({
                "fault_id":       fid,
                "fault_type":     ftype,
                "source":         source,
                "error":          error,
                "correct_action": correct,
                "approach":       name,
                "decision":       result,
                "is_correct":     is_correct,
                "latency":        latency,
            })

        print()

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "═" * 62)
    print("  EVALUATION SUMMARY")
    print("═" * 62)

    summary = (
        df.groupby("approach")
        .agg(
            accuracy    = ("is_correct", "mean"),
            avg_latency = ("latency",    "mean"),
            n           = ("is_correct", "count"),
        )
        .sort_values("accuracy", ascending=False)
    )

    print(f"\n  {'Approach':<14} {'Accuracy':>9} {'Avg Latency':>13} {'N':>5}")
    print("  " + "─" * 46)
    for name, row in summary.iterrows():
        bar = "█" * int(row["accuracy"] * 20)
        print(f"  {name:<14} {row['accuracy']:>8.0%}  "
              f"{row['avg_latency']:>10.3f}s  {int(row['n']):>4}  {bar}")

    # Per fault type — LLM+RAG
    print("\n📊 ACCURACY BY FAULT TYPE  (LLM+RAG)")
    print("─" * 50)
    llm_rag_df = df[df["approach"] == "LLM+RAG"]
    if len(llm_rag_df) > 0:
        by_fault = (
            llm_rag_df.groupby("fault_type")["is_correct"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        for ftype, row in by_fault.iterrows():
            bar = "█" * int(row["mean"] * 20)
            print(f"  {ftype:<30} {row['mean']:.0%}  {bar}  (n={int(row['count'])})")

    # Library vs novel
    print("\n📊 LIBRARY vs LLM-GENERATED  (LLM+RAG)")
    print("─" * 50)
    for source in ["library", "llm_generated", "library_fallback"]:
        subset = llm_rag_df[llm_rag_df["source"] == source]
        if len(subset) > 0:
            acc = subset["is_correct"].mean()
            bar = "█" * int(acc * 20)
            print(f"  {source:<22} {acc:.0%}  {bar}  (n={len(subset)})")

    # All approaches on novel faults only
    novel_df = df[df["source"].isin(["llm_generated", "library_fallback"])]
    if len(novel_df) > 0:
        print("\n📊 NOVEL FAULTS ONLY  (all approaches)")
        print("─" * 50)
        novel_summary = (
            novel_df.groupby("approach")["is_correct"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        for name, row in novel_summary.iterrows():
            bar = "█" * int(row["mean"] * 20)
            print(f"  {name:<14} {row['mean']:.0%}  {bar}  (n={int(row['count'])})")


def save_results(df: pd.DataFrame,
                 path: str = "evaluation/baseline_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Results saved to {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = evaluate_all(include_llm_faults=3)
    print_summary(df)
    save_results(df)