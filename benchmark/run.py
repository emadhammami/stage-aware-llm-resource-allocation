from __future__ import annotations

import argparse
from typing import cast

from agent.state import MethodName
from benchmark.config import ExperimentConfig
from benchmark.quixbugs import QuixBugsBenchmark
from benchmark.results import completed_keys
from benchmark.runner import run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["single_shot", "pec", "pevc", "evidence_gated"])
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--tasks", required=True, help="'all' or comma-separated task ids")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = QuixBugsBenchmark()
    config = ExperimentConfig.load()
    tasks = benchmark.discover_tasks() if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    done = completed_keys()
    for task in tasks:
        key = (task, args.method, args.budget, args.repetition)
        if key in done and not args.force:
            print(f"skip completed {task} {args.method} {args.budget} run{args.repetition}")
            continue
        print(f"run {task} {args.method} {args.budget} run{args.repetition}")
        run_one(
            task_id=task,
            method=cast(MethodName, args.method),
            budget=args.budget,
            repetition=args.repetition,
            is_pilot=args.pilot,
            benchmark=benchmark,
            experiment_config=config,
        )


if __name__ == "__main__":
    main()
