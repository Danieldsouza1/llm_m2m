"""
evaluation/logger.py
---------------------
Improved logger — now records fault_type and correct_action
so the evaluator can score every approach objectively.
"""

import csv, os, time

LOG_FILE = "evaluation/results.csv"

COLUMNS = [
    "timestamp",
    "task",
    "agent",
    "status",
    "fault_type",
    "error",
    "decision",
    "correct_action",
    "is_correct",    # 1 if decision == correct_action, 0 otherwise
    "latency",
]


def init_logger(path: str = LOG_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
    print(f"📋 Logger initialised: {path}")


def log_event(
    task: str,
    agent: str,
    status: str,
    error: str,
    decision: str,
    correct_action: str,
    fault_type: str,
    start_time: float,
    path: str = LOG_FILE,
):
    latency = round(time.time() - start_time, 3)

    # Score: only meaningful for FAILURE rows that have a correct_action
    if status == "FAILURE" and correct_action:
        is_correct = 1 if decision.upper() == correct_action.upper() else 0
    else:
        is_correct = ""   # N/A for SUCCESS rows

    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        task,
        agent,
        status,
        fault_type,
        error,
        decision,
        correct_action,
        is_correct,
        latency,
    ]

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"📝 Logged: {task} | {status} | decision={decision} | correct={correct_action} | latency={latency}s")
