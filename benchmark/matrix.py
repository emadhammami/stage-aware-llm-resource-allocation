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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--core", action="store_true")
    group.add_argument("--pilot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repetition", type=int, default=1)
    return parser.parse_args()


def execute_plan(
    plan,
    *,
    repetition: int,
    is_pilot: bool,
    force: bool,
    benchmark: QuixBugsBenchmark,
    config: ExperimentConfig,
    completed_csv: str = "results/runs.csv",
    run_one_fn=run_one,
) -> None:
    done = completed_keys(completed_csv)
    for task, method, budget in plan:
        key = (task, method, budget, repetition)
        if key in done and not force:
            print(f"skip completed {task} {method} {budget} run{repetition}")
            continue
        print(f"run {task} {method} {budget} run{repetition} pilot={is_pilot}")
        run_one_fn(
            task_id=task,
            method=cast(MethodName, method),
            budget=int(budget),
            repetition=repetition,
            is_pilot=is_pilot,
            benchmark=benchmark,
            experiment_config=config,
        )


def main() -> None:
    args = parse_args()
    benchmark = QuixBugsBenchmark()
    config = ExperimentConfig.load()
    if args.pilot:
        plan = [(task, config.pilot_method, config.pilot_budget) for task in config.pilot_tasks]
        is_pilot = True
    else:
        tasks = benchmark.discover_tasks()
        main_budget = config.main_comparison_budget
        lower_budgets = [budget for budget in config.required_budgets if budget != main_budget]
        methods = ["single_shot", "pec", "pevc", "evidence_gated"]
        plan = [(task, method, main_budget) for task in tasks for method in methods]
        plan += [(task, "evidence_gated", budget) for task in tasks for budget in lower_budgets]
        is_pilot = False
    execute_plan(
        plan,
        repetition=args.repetition,
        is_pilot=is_pilot,
        force=args.force,
        benchmark=benchmark,
        config=config,
    )


if __name__ == "__main__":
    main()
