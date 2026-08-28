from __future__ import annotations

from typing import Any

from benchmark.hotpotqa import HotpotTask
from benchmark.hotpotqa_eval import evaluate_answer

EVALUATION_VERSION = "confirmatory-v1"


def _run_end(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("run_end")
    return value if isinstance(value, dict) else {}


def _normal_completion(result: dict[str, Any]) -> bool:
    return bool(_run_end(result).get("normal_completion", False))


def _zero_budget_violations(result: dict[str, Any]) -> bool:
    return int(_run_end(result).get("budget_violation_count", 0)) == 0


def score_hotpot_result(result: dict[str, Any], task: HotpotTask) -> dict[str, Any]:
    candidate = result.get("candidate_answer")
    prediction = candidate if isinstance(candidate, str) and candidate.strip() else None
    gold = str(task.evaluation_gold()["answer"])
    metrics = evaluate_answer(prediction, gold)

    exact_match = float(metrics["exact_match"])
    token_f1 = float(metrics["token_f1"])
    answer_available = bool(metrics["answer_available"])
    normal_completion = _normal_completion(result)
    answer_parse_ok = bool(result.get("answer_parse_ok", False))
    verification_parse_ok = bool(result.get("final_verification_parse_ok", False))
    verification_verdict = result.get("final_verification_verdict")
    zero_budget_violations = _zero_budget_violations(result)

    end_to_end_exact_match = exact_match if normal_completion else 0.0
    end_to_end_token_f1 = token_f1 if normal_completion else 0.0
    reliable_correct = bool(
        exact_match == 1.0
        and normal_completion
        and answer_parse_ok
        and verification_parse_ok
        and zero_budget_violations
    )
    verification_supported_correct = bool(
        reliable_correct and verification_verdict == "SUFFICIENT"
    )

    return {
        "evaluation_version": EVALUATION_VERSION,
        "answer_available": answer_available,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "end_to_end_exact_match": end_to_end_exact_match,
        "end_to_end_token_f1": end_to_end_token_f1,
        "reliable_correct": reliable_correct,
        "verification_supported_correct": verification_supported_correct,
    }


def score_quixbugs_result(result: dict[str, Any]) -> dict[str, Any]:
    validation = result.get("validation")
    validation_success = (
        bool(validation.get("success", False))
        if isinstance(validation, dict)
        else False
    )
    functional_correct = bool(result.get("functional_correct", validation_success))
    normal_completion = _normal_completion(result)
    zero_budget_violations = _zero_budget_violations(result)
    stages = result.get("stages")
    critic_completed = bool(
        isinstance(stages, list)
        and any(
            isinstance(stage, dict) and stage.get("stage_id") == "critic"
            for stage in stages
        )
    )

    end_to_end_functional_correct = bool(functional_correct and normal_completion)
    reliable_correct = bool(
        end_to_end_functional_correct
        and critic_completed
        and zero_budget_violations
    )

    return {
        "evaluation_version": EVALUATION_VERSION,
        "functional_correct": functional_correct,
        "end_to_end_functional_correct": end_to_end_functional_correct,
        "critic_completed": critic_completed,
        "reliable_correct": reliable_correct,
    }
