"""
evaluation/metrics.py
----------------------
Improved metrics — computes per-approach and per-fault-type accuracy.
"""

import pandas as pd
import os


def compute_metrics(file: str = "evaluation/results.csv"):
    if not os.path.exists(file):
        print(f"❌ Results file not found: {file}")
        return

    df = pd.read_csv(file)

    if df.empty:
        print("❌ Results file is empty")
        return

    total   = len(df)
    success = len(df[df["status"] == "SUCCESS"])
    failure = len(df[df["status"] == "FAILURE"])

    failure_df = df[df["status"] == "FAILURE"].copy()

    # Only score rows that have a ground truth
    scoreable = failure_df[failure_df["correct_action"].notna() & (failure_df["correct_action"] != "")]

    if len(scoreable) > 0:
        accuracy = scoreable["is_correct"].astype(float).mean()
    else:
        accuracy = float("nan")

    avg_latency = df["latency"].mean()

    print("\n📊 OVERALL METRICS")
    print("─" * 40)
    print(f"  Total events:       {total}")
    print(f"  Successes:          {success}  ({success/total:.0%})")
    print(f"  Failures:           {failure}  ({failure/total:.0%})")
    print(f"  Recovery accuracy:  {accuracy:.0%}" if not pd.isna(accuracy) else "  Recovery accuracy:  N/A")
    print(f"  Average latency:    {avg_latency:.3f}s")

    # Per fault type breakdown
    if "fault_type" in failure_df.columns and len(scoreable) > 0:
        print("\n📊 ACCURACY BY FAULT TYPE")
        print("─" * 40)
        by_fault = (
            scoreable.groupby("fault_type")["is_correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .sort_values("accuracy", ascending=False)
        )
        for fault, row in by_fault.iterrows():
            bar = "█" * int(row["accuracy"] * 20)
            print(f"  {fault:<25} {row['accuracy']:.0%}  {bar}  (n={int(row['n'])})")

    # Decision distribution
    if len(failure_df) > 0:
        print("\n📊 DECISION DISTRIBUTION (failures only)")
        print("─" * 40)
        dist = failure_df["decision"].value_counts()
        for action, count in dist.items():
            print(f"  {action:<12} {count}")

    return df


if __name__ == "__main__":
    compute_metrics()
