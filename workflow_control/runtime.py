from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent.prompts import critic_prompt, executor_prompt, planner_prompt, retry_prompt
from agent.state import ValidationResult
from benchmark.hotpotqa import DeterministicRetriever, HotpotTask
from benchmark.patcher import apply_function_patch
from benchmark.quixbugs import QuixBugsBenchmark
from workflow_control.backends import ModelBackend
from workflow_control.controller import BudgetController
from workflow_control.parsing import parse_final_answer, parse_verdict
from workflow_control.prompts import (
    hotpot_answer_prompt,
    hotpot_plan_prompt,
    hotpot_revision_prompt,
    hotpot_verify_prompt,
)


@dataclass(frozen=True)
class StageOutput:
    stage_id: str
    text: str
    usage: dict[str, Any]
    model_metadata: dict[str, Any]


class StructuralShortfallError(RuntimeError):
    """Expected scientific terminal condition when a stage minimum cannot fit."""

    def __init__(self, *, call_id: str, stage_id: str, remaining: int, required_minimum: int) -> None:
        self.call_id = call_id
        self.stage_id = stage_id
        self.remaining = remaining
        self.required_minimum = required_minimum
        super().__init__(
            f"structural shortfall before {call_id}: remaining={remaining}, "
            f"required={required_minimum}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "stage_id": self.stage_id,
            "remaining": self.remaining,
            "required_minimum": self.required_minimum,
        }


def call_stage(
    backend: ModelBackend,
    controller: BudgetController,
    *,
    call_id: str,
    stage_id: str,
    prompt: str,
) -> StageOutput:
    messages = [{"role": "user", "content": prompt}]
    exact_prompt, _, _ = backend.count_prompt(messages)
    controller.materialize_future_prompt(stage_id, exact_prompt)
    allocation = controller.prepare_call(call_id, stage_id, exact_prompt)
    if not allocation.admitted:
        raise StructuralShortfallError(
            call_id=call_id,
            stage_id=stage_id,
            remaining=allocation.budget_before,
            required_minimum=allocation.required_minimum,
        )
    text, usage, metadata = backend.generate(messages, allocation.selected_max_output)
    controller.finish_call(
        call_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        provider={
            "native_finish_reason": usage.finish_reason,
            "mapped_finish_class": usage.mapped_finish_class,
            "latency_seconds": usage.latency_seconds,
            "model_metadata": metadata.to_dict(),
        },
    )
    return StageOutput(stage_id, text, asdict(usage), metadata.to_dict())


def _planner_fields(text: str) -> tuple[str | None, str]:
    target = None
    hypothesis = text.strip()
    for line in text.splitlines():
        if line.startswith("TARGET_FUNCTION:"):
            target = line.partition(":")[2].strip() or None
        elif line.startswith("HYPOTHESIS:"):
            hypothesis = line.partition(":")[2].strip()
    return target, hypothesis


def _validate_patch(
    benchmark: QuixBugsBenchmark,
    task_id: str,
    patch: str,
    target_function: str | None,
) -> ValidationResult:
    with benchmark.task_worktree(task_id) as environment:
        patch_record = apply_function_patch(
            environment.program_path,
            patch,
            function_name=target_function,
        )
        if not patch_record.applied:
            return ValidationResult(
                success=False,
                error_category="patch_error",
                stderr=patch_record.error or "",
                failing_test_info=patch_record.error or "patch could not be applied",
            )
        if not patch_record.syntax_valid:
            return ValidationResult(
                success=False,
                error_category="syntax_error",
                stderr=patch_record.error or "",
                failing_test_info=patch_record.error or "invalid Python syntax",
            )
        return benchmark.run_tests(environment)


def run_code_workflow(
    *,
    backend: ModelBackend,
    controller: BudgetController,
    benchmark: QuixBugsBenchmark,
    task_id: str,
) -> dict[str, Any]:
    code = benchmark.load_buggy_code(task_id)
    outputs: list[StageOutput] = []
    plan = call_stage(
        backend,
        controller,
        call_id="planner:1",
        stage_id="planner",
        prompt=planner_prompt(task_id, code),
    )
    outputs.append(plan)
    target, hypothesis = _planner_fields(plan.text)
    attempt = call_stage(
        backend,
        controller,
        call_id="executor_1:1",
        stage_id="executor_1",
        prompt=executor_prompt(task_id, code, hypothesis, controller.remaining),
    )
    outputs.append(attempt)
    evidence = _validate_patch(benchmark, task_id, attempt.text, target)
    patch = attempt.text
    if evidence.success:
        controller.release_unreachable({"planner", "executor_1", "critic"})
    else:
        retry = call_stage(
            backend,
            controller,
            call_id="executor_2:1",
            stage_id="executor_2",
            prompt=retry_prompt(
                task_id,
                code,
                attempt.text,
                evidence,
                hypothesis,
                controller.remaining,
            ),
        )
        outputs.append(retry)
        patch = retry.text
        evidence = _validate_patch(benchmark, task_id, patch, target)
    critic = call_stage(
        backend,
        controller,
        call_id="critic:1",
        stage_id="critic",
        prompt=critic_prompt(task_id, code, hypothesis, patch, evidence),
    )
    outputs.append(critic)
    return {
        "task_id": task_id,
        "validation": evidence.model_dump(),
        "functional_correct": bool(evidence.success),
        "critic": critic.text,
        "stages": [asdict(output) for output in outputs],
        "run_end": controller.finalize(),
    }


def run_hotpot_workflow(
    *,
    backend: ModelBackend,
    controller: BudgetController,
    task: HotpotTask,
) -> dict[str, Any]:
    outputs: list[StageOutput] = []

    def invalid_response(
        *,
        stage_id: str,
        raw_answer: str | None,
        parsed_answer: str | None,
        answer_parse_ok: bool,
        final_verification_stage: str | None,
        initial_verifier_verdict: str | None = None,
        recovery_documents: int = 0,
    ) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "terminal_status": "invalid_response",
            "classification": "scientific",
            "parse_failure_stage": stage_id,
            "candidate_answer_raw": raw_answer,
            "candidate_answer": parsed_answer,
            "answer_parse_ok": answer_parse_ok,
            "initial_verifier_verdict": initial_verifier_verdict,
            "final_verification_stage": final_verification_stage,
            "final_verification_parse_ok": False,
            "final_verification_verdict": None,
            "recovery_documents": recovery_documents,
            "stages": [asdict(output) for output in outputs],
            "run_end": controller.finalize(normal_completion=False),
        }

    plan = call_stage(
        backend,
        controller,
        call_id="plan:1",
        stage_id="plan",
        prompt=hotpot_plan_prompt(task),
    )
    outputs.append(plan)

    retriever = DeterministicRetriever(task.documents)
    ranked = retriever.ranked(f"{task.question} {plan.text}")
    evidence = tuple(ranked[:2])

    answer = call_stage(
        backend,
        controller,
        call_id="answer:1",
        stage_id="answer",
        prompt=hotpot_answer_prompt(task, evidence),
    )
    outputs.append(answer)
    parsed_answer = parse_final_answer(answer.text)
    if parsed_answer is None:
        return invalid_response(
            stage_id="answer",
            raw_answer=answer.text,
            parsed_answer=None,
            answer_parse_ok=False,
            final_verification_stage=None,
        )

    verifier = call_stage(
        backend,
        controller,
        call_id="verifier:1",
        stage_id="verifier",
        prompt=hotpot_verify_prompt(task, evidence, answer.text),
    )
    outputs.append(verifier)
    verifier_verdict = parse_verdict(verifier.text)
    if verifier_verdict is None:
        return invalid_response(
            stage_id="verifier",
            raw_answer=answer.text,
            parsed_answer=parsed_answer,
            answer_parse_ok=True,
            final_verification_stage="verifier",
        )

    final_answer_raw = answer.text
    final_answer = parsed_answer
    final_verification_stage = "verifier"
    final_verification_verdict = verifier_verdict
    recovery_documents = 0

    if verifier_verdict == "SUFFICIENT":
        controller.release_unreachable({"plan", "answer", "verifier"})
    else:
        seen = {document.title for document in evidence}
        additional = retriever.next_unseen_many(
            f"{task.question} {plan.text}",
            seen,
            limit=2,
        )
        if len(additional) != 2:
            raise RuntimeError("fewer than two unseen HotpotQA recovery documents remain")
        recovery_documents = len(additional)
        expanded = evidence + additional

        revision = call_stage(
            backend,
            controller,
            call_id="revise:1",
            stage_id="revise",
            prompt=hotpot_revision_prompt(task, expanded, answer.text),
        )
        outputs.append(revision)
        parsed_revision = parse_final_answer(revision.text)
        if parsed_revision is None:
            return invalid_response(
                stage_id="revise",
                raw_answer=revision.text,
                parsed_answer=None,
                answer_parse_ok=False,
                final_verification_stage=None,
                initial_verifier_verdict=verifier_verdict,
                recovery_documents=recovery_documents,
            )

        final_answer_raw = revision.text
        final_answer = parsed_revision
        terminal = call_stage(
            backend,
            controller,
            call_id="terminal_verifier:1",
            stage_id="terminal_verifier",
            prompt=hotpot_verify_prompt(task, expanded, revision.text),
        )
        outputs.append(terminal)
        terminal_verdict = parse_verdict(terminal.text)
        if terminal_verdict is None:
            return invalid_response(
                stage_id="terminal_verifier",
                raw_answer=revision.text,
                parsed_answer=parsed_revision,
                answer_parse_ok=True,
                final_verification_stage="terminal_verifier",
                initial_verifier_verdict=verifier_verdict,
                recovery_documents=recovery_documents,
            )
        final_verification_stage = "terminal_verifier"
        final_verification_verdict = terminal_verdict

    return {
        "task_id": task.task_id,
        "candidate_answer_raw": final_answer_raw,
        "candidate_answer": final_answer,
        "answer_parse_ok": True,
        "initial_verifier_verdict": verifier_verdict,
        "final_verification_stage": final_verification_stage,
        "final_verification_parse_ok": True,
        "final_verification_verdict": final_verification_verdict,
        "recovery_documents": recovery_documents,
        "stages": [asdict(output) for output in outputs],
        "run_end": controller.finalize(),
    }
