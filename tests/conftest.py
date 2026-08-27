from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest

from benchmark.quixbugs import TaskEnvironment


@pytest.fixture
def sample_source(tmp_path: Path) -> Path:
    path = tmp_path / "gcd.py"
    path.write_text(
        "def gcd(a, b):\n"
        "    if b == 0:\n"
        "        return a\n"
        "    return gcd(a, a % b)\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_quixbugs(tmp_path: Path):
    root = tmp_path / "QuixBugs"
    programs = root / "python_programs"
    tests = root / "python_testcases"
    programs.mkdir(parents=True)
    tests.mkdir()
    (programs / "gcd.py").write_text(
        "def gcd(a, b):\n"
        "    if b == 0:\n"
        "        return a\n"
        "    return gcd(a, a % b)\n",
        encoding="utf-8",
    )
    (tests / "test_gcd.py").write_text(
        "from gcd import gcd\n\n"
        "def test_gcd():\n"
        "    assert gcd(8, 4) == 4\n"
        "    assert gcd(9, 6) == 3\n",
        encoding="utf-8",
    )

    class FakeBenchmark:
        local_path = root
        programs_dir = "python_programs"
        tests_dir = "python_testcases"
        timeout_seconds = 20

        def load_buggy_code(self, task_id: str) -> str:
            return (programs / f"{task_id}.py").read_text(encoding="utf-8")

        def benchmark_commit(self) -> str:
            return "fake-commit"

        def task_worktree(self, task_id: str):
            @contextmanager
            def manager():
                copy_root = tmp_path / f"copy_{task_id}"
                if copy_root.exists():
                    shutil.rmtree(copy_root)
                shutil.copytree(root, copy_root)
                yield TaskEnvironment(
                    root=copy_root,
                    task_id=task_id,
                    program_path=copy_root / "python_programs" / f"{task_id}.py",
                    test_path=copy_root / "python_testcases" / f"test_{task_id}.py",
                )

            return manager()

        def run_tests(self, env):
            from benchmark.quixbugs import QuixBugsBenchmark

            bench = QuixBugsBenchmark.__new__(QuixBugsBenchmark)
            bench.programs_dir = "python_programs"
            bench.timeout_seconds = 20
            return QuixBugsBenchmark.run_tests(bench, env)

    return FakeBenchmark()
