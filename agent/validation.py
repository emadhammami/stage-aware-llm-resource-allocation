from __future__ import annotations

from pathlib import Path

from agent.state import RepairState, ValidationResult
from benchmark.patcher import apply_function_patch
from benchmark.quixbugs import QuixBugsBenchmark


def validate_candidate(state: RepairState, benchmark: QuixBugsBenchmark) -> None:
    patch_code = state.executor_outputs[-1].proposed_code if state.executor_outputs else ""
    with benchmark.task_worktree(state.task_id) as task_env:
        source_path = task_env.program_path
        original = Path(source_path).read_text(encoding="utf-8")
        function_name = state.planner.target_function if state.planner else None
        patch_record = apply_function_patch(source_path, patch_code, function_name=function_name)
        state.patch = patch_record
        if not patch_record.applied:
            result = ValidationResult(
                success=False,
                stdout="",
                stderr=patch_record.error or "",
                return_code=None,
                error_category="patch_error",
                failing_test_info=patch_record.error or "Patch could not be applied.",
            )
            state.validations.append(result)
            state.final_error_category = result.error_category
            state.add_event("validation", patch_applied=False)
            return
        if not patch_record.syntax_valid:
            result = ValidationResult(
                success=False,
                stdout="",
                stderr=patch_record.error or "",
                return_code=None,
                error_category="syntax_error",
                failing_test_info=patch_record.error or "Patched source is not valid Python.",
            )
            state.validations.append(result)
            state.final_error_category = result.error_category
            state.add_event("validation", patch_applied=True, syntax_valid=False)
            Path(source_path).write_text(original, encoding="utf-8")
            return
        result = benchmark.run_tests(task_env)
        state.validations.append(result)
        state.final_error_category = result.error_category
        state.add_event(
            "validation",
            patch_applied=True,
            syntax_valid=True,
            success=result.success,
            error_category=result.error_category,
        )

