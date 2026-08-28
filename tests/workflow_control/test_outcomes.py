from __future__ import annotations

from benchmark.hotpotqa import HotpotTask
from benchmark.outcomes import score_hotpot_result, score_quixbugs_result

TASK = HotpotTask(
    task_id="fixture-hotpot",
    question="In which month did the event occur?",
    question_type="bridge",
    level="easy",
    documents=(),
    _gold_answer="October",
    _gold_supporting_facts=(),
)


def test_hotpot_correct_normal_run_is_reliable() -> None:
    result = {
        "candidate_answer": "October",
        "answer_parse_ok": True,
        "final_verification_parse_ok": True,
        "final_verification_verdict": "SUFFICIENT",
        "run_end": {"normal_completion": True, "budget_violation_count": 0},
    }
    metrics = score_hotpot_result(result, TASK)
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["end_to_end_exact_match"] == 1.0
    assert metrics["reliable_correct"] is True
    assert metrics["verification_supported_correct"] is True


def test_hotpot_abnormal_run_keeps_raw_but_zeros_end_to_end() -> None:
    result = {
        "candidate_answer": "October",
        "answer_parse_ok": True,
        "final_verification_parse_ok": False,
        "run_end": {"normal_completion": False, "budget_violation_count": 0},
    }
    metrics = score_hotpot_result(result, TASK)
    assert metrics["exact_match"] == 1.0
    assert metrics["end_to_end_exact_match"] == 0.0
    assert metrics["end_to_end_token_f1"] == 0.0
    assert metrics["reliable_correct"] is False


def test_hotpot_verifier_disagreement_is_reported_separately() -> None:
    result = {
        "candidate_answer": "October",
        "answer_parse_ok": True,
        "final_verification_parse_ok": True,
        "final_verification_verdict": "INSUFFICIENT",
        "run_end": {"normal_completion": True, "budget_violation_count": 0},
    }
    metrics = score_hotpot_result(result, TASK)
    assert metrics["reliable_correct"] is True
    assert metrics["verification_supported_correct"] is False


def test_quixbugs_functional_success_requires_normal_run_and_critic_for_reliability() -> None:
    good = {
        "validation": {"success": True},
        "functional_correct": True,
        "stages": [{"stage_id": "critic"}],
        "run_end": {"normal_completion": True, "budget_violation_count": 0},
    }
    bad = {
        "validation": {"success": True},
        "functional_correct": True,
        "stages": [],
        "run_end": {"normal_completion": False, "budget_violation_count": 0},
    }
    good_metrics = score_quixbugs_result(good)
    bad_metrics = score_quixbugs_result(bad)
    assert good_metrics["end_to_end_functional_correct"] is True
    assert good_metrics["critic_completed"] is True
    assert good_metrics["reliable_correct"] is True
    assert bad_metrics["functional_correct"] is True
    assert bad_metrics["end_to_end_functional_correct"] is False
    assert bad_metrics["reliable_correct"] is False
