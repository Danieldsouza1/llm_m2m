"""
run_eval.py
-----------
Runs the full live grid system (with MQTT agents) against a dynamic
fault schedule, then prints metrics.

Usage:
  python run_eval.py --runs 5 --llm-ratio 0.5

  --runs:      number of evaluation runs (default 5)
  --llm-ratio: probability of LLM-generated faults (default 0.5)
               0.0 = all library,  1.0 = all LLM-generated
"""

import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import argparse, importlib, time
import grid_controller   # import the module so we can set fault_schedule
from fault_generator import get_random_fault
from evaluation.metrics import compute_metrics


def build_fault_schedule(llm_ratio: float) -> dict:
    """
    Build a fault schedule for one run.
    Steps: 0=generate_power, 1=transmit_power, 2=distribute_power
    We always inject a fault at step 1 (the transmission step) to simulate
    the same scenario as the original project, but now the fault is dynamic.
    """
    fault = get_random_fault(fault_id="RUN_FAULT", llm_ratio=llm_ratio)
    print(f"\n💉 Fault for this run: [{fault['fault_type']}] {fault['error_message']}")
    print(f"   Correct action: {fault['correct_action']}  |  Source: {fault['source']}")
    return {1: fault}   # inject at step 1 (transmit_power)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",      type=int,   default=5,   help="Number of evaluation runs")
    parser.add_argument("--llm-ratio", type=float, default=0.5, help="LLM fault generation probability")
    args = parser.parse_args()

    print(f"🚀 Starting evaluation: {args.runs} runs, llm_ratio={args.llm_ratio}")

    for i in range(args.runs):
        print(f"\n{'='*50}")
        print(f"RUN {i+1}/{args.runs}")
        print("=" * 50)

        # Set the fault schedule on the grid_controller module
        schedule = build_fault_schedule(args.llm_ratio)
        grid_controller.fault_schedule = schedule

        # Reset controller state for this run
        grid_controller.current_step = 0
        grid_controller.retry_count  = 0

        # Re-initialise logger and trigger the pipeline
        grid_controller.init_logger()
        grid_controller.trigger_next()
        grid_controller.client.loop_forever()

        time.sleep(2)

    # Final metrics across all runs
    print("\n" + "=" * 50)
    print("FINAL METRICS ACROSS ALL RUNS")
    print("=" * 50)
    compute_metrics()


if __name__ == "__main__":
    main()
