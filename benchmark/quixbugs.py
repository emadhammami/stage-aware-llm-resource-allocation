from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent.state import ErrorCategory, ValidationResult


@dataclass(frozen=True)
class TaskEnvironment:
    root: Path
    task_id: str
    program_path: Path
    test_path: Path


class QuixBugsBenchmark:
    def __init__(self, config_path: str | Path = "configs/benchmark.yaml") -> None:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        quix = config["quixbugs"]
        self.repo_url = quix["repo_url"]
        self.commit_sha = quix["commit_sha"]
        self.local_path = Path(quix["local_path"])
        self.programs_dir = quix["python_programs_dir"]
        self.tests_dir = quix["python_tests_dir"]
        self.timeout_seconds = int(quix.get("timeout_seconds", 10))
        self.tasks = list(quix["tasks"])

    def setup(self) -> None:
        if not self.local_path.exists():
            subprocess.run(["git", "clone", self.repo_url, str(self.local_path)], check=True)
        subprocess.run(["git", "fetch", "--all"], cwd=self.local_path, check=True)
        subprocess.run(["git", "checkout", self.commit_sha], cwd=self.local_path, check=True)

    def benchmark_commit(self) -> str:
        if not self.local_path.exists():
            return self.commit_sha
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.local_path,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout.strip() or self.commit_sha

    def discover_tasks(self) -> list[str]:
        if len(self.tasks) != 40:
            raise RuntimeError(f"Configured QuixBugs task allowlist must contain 40 tasks, got {len(self.tasks)}")
        missing: list[str] = []
        for task_id in self.tasks:
            if not (self.local_path / self.programs_dir / f"{task_id}.py").exists():
                missing.append(f"{self.programs_dir}/{task_id}.py")
            if not (self.local_path / self.tests_dir / f"test_{task_id}.py").exists():
                missing.append(f"{self.tests_dir}/test_{task_id}.py")
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise RuntimeError(f"QuixBugs checkout does not match configured task set:\n{details}")
        return list(self.tasks)

    def load_buggy_code(self, task_id: str) -> str:
        self._validate_configured_task(task_id)
        return (self.local_path / self.programs_dir / f"{task_id}.py").read_text(encoding="utf-8")

    @contextlib.contextmanager
    def task_worktree(self, task_id: str):
        source_program = self.local_path / self.programs_dir / f"{task_id}.py"
        source_test = self._test_path(task_id)
        with tempfile.TemporaryDirectory(prefix=f"quixbugs_{task_id}_") as temp:
            root = Path(temp)
            shutil.copytree(self.local_path, root / "QuixBugs", ignore=shutil.ignore_patterns(".git"))
            copied_root = root / "QuixBugs"
            yield TaskEnvironment(
                root=copied_root,
                task_id=task_id,
                program_path=copied_root / self.programs_dir / source_program.name,
                test_path=copied_root / source_test.relative_to(self.local_path),
            )

    def _test_path(self, task_id: str) -> Path:
        self._validate_configured_task(task_id)
        test_root = self.local_path / self.tests_dir
        candidate = test_root / f"test_{task_id}.py"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No official QuixBugs Python test found for {task_id}: {candidate}")

    def _validate_configured_task(self, task_id: str) -> None:
        if task_id not in self.tasks:
            raise ValueError(f"{task_id!r} is not in the configured QuixBugs task allowlist")

    def run_tests(self, env: TaskEnvironment) -> ValidationResult:
        start = time.perf_counter()
        command = [sys.executable, "-m", "pytest", str(env.test_path)]
        env_vars = os.environ.copy()
        env_vars["PYTHONPATH"] = str(env.root / self.programs_dir) + os.pathsep + env_vars.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                command,
                cwd=env.root,
                env=env_vars,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
            runtime = time.perf_counter() - start
            passed, failed, total = parse_pytest_counts(proc.stdout + "\n" + proc.stderr)
            category = classify_test_result(proc.returncode, proc.stdout, proc.stderr, False)
            return ValidationResult(
                success=proc.returncode == 0 and total > 0,
                tests_passed=passed,
                tests_failed=failed,
                tests_total=total,
                stdout=proc.stdout[-32768:],
                stderr=proc.stderr[-32768:],
                return_code=proc.returncode,
                timed_out=False,
                runtime_seconds=runtime,
                error_category=category,
                failing_test_info=extract_failing_info(proc.stdout + "\n" + proc.stderr),
            )
        except subprocess.TimeoutExpired as exc:
            runtime = time.perf_counter() - start
            return ValidationResult(
                success=False,
                stdout=(exc.stdout or "")[-32768:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-32768:] if isinstance(exc.stderr, str) else "",
                return_code=None,
                timed_out=True,
                runtime_seconds=runtime,
                error_category="timeout",
                failing_test_info="pytest timed out",
            )


def parse_pytest_counts(output: str) -> tuple[int, int, int]:
    failed = passed = 0
    match = re.search(r"(\d+)\s+failed", output)
    if match:
        failed = int(match.group(1))
    match = re.search(r"(\d+)\s+passed", output)
    if match:
        passed = int(match.group(1))
    total = failed + passed
    return passed, failed, total


def classify_test_result(return_code: int, stdout: str, stderr: str, timed_out: bool) -> ErrorCategory:
    if timed_out:
        return "timeout"
    text = f"{stdout}\n{stderr}"
    if return_code == 0:
        return "none"
    if "SyntaxError" in text:
        return "syntax_error"
    if "ImportError" in text or "ModuleNotFoundError" in text:
        return "import_error"
    if "AssertionError" in text or "failed" in text:
        return "assertion_failure"
    if "Error" in text or "Exception" in text:
        return "runtime_error"
    return "other"


def extract_failing_info(output: str, limit: int = 4000) -> str:
    markers = ["=================================== FAILURES", "==================================== ERRORS"]
    start = 0
    for marker in markers:
        index = output.find(marker)
        if index != -1:
            start = index
            break
    excerpt = output[start:]
    summary_index = excerpt.find("=========================== short test summary info")
    if summary_index != -1:
        excerpt = excerpt[:summary_index]
    if not excerpt.strip():
        excerpt = output
    excerpt = excerpt.strip()
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[-limit:]


def runtime_fingerprint() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
