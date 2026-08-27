import shutil
from pathlib import Path

import pytest
import yaml

from benchmark.quixbugs import (
    QuixBugsBenchmark,
    classify_test_result,
    extract_failing_info,
    parse_pytest_counts,
)


def test_parse_pytest_counts():
    assert parse_pytest_counts("2 failed, 3 passed in 0.10s") == (3, 2, 5)


def test_classify_assertion_failure():
    category = classify_test_result(1, "FAILED test_x.py::test_y - AssertionError", "", False)
    assert category == "assertion_failure"


def test_failure_excerpt_keeps_assertion_context():
    output = """
=================================== FAILURES ===================================
FAILED python_testcases/test_gcd.py::test_gcd
>       assert gcd(9, 6) == 3
E       assert 0 == 3
E        +  where 0 = gcd(9, 6)
=========================== short test summary info ===========================
"""
    excerpt = extract_failing_info(output)
    assert "assert gcd(9, 6) == 3" in excerpt
    assert "assert 0 == 3" in excerpt


def test_configured_task_allowlist_discovers_exactly_40_and_excludes_helpers(tmp_path: Path):
    tasks = [f"task_{index}" for index in range(40)]
    root = tmp_path / "QuixBugs"
    programs = root / "python_programs"
    tests = root / "python_testcases"
    programs.mkdir(parents=True)
    tests.mkdir()
    for task in tasks:
        (programs / f"{task}.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (tests / f"test_{task}.py").write_text("def test_f():\n    assert True\n", encoding="utf-8")
    (programs / "node.py").write_text("class Node: pass\n", encoding="utf-8")
    (programs / "gcd_test.py").write_text("def helper(): pass\n", encoding="utf-8")
    config = {
        "quixbugs": {
            "repo_url": "unused",
            "commit_sha": "fake",
            "local_path": str(root),
            "python_programs_dir": "python_programs",
            "python_tests_dir": "python_testcases",
            "timeout_seconds": 10,
            "tasks": tasks,
        }
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    discovered = QuixBugsBenchmark(config_path).discover_tasks()
    assert len(discovered) == 40
    assert discovered == tasks
    assert "node" not in discovered
    assert "gcd_test" not in discovered


def test_configured_task_allowlist_fails_when_checkout_mismatches(tmp_path: Path):
    tasks = [f"task_{index}" for index in range(40)]
    root = tmp_path / "QuixBugs"
    (root / "python_programs").mkdir(parents=True)
    (root / "python_testcases").mkdir()
    config = {
        "quixbugs": {
            "repo_url": "unused",
            "commit_sha": "fake",
            "local_path": str(root),
            "python_programs_dir": "python_programs",
            "python_tests_dir": "python_testcases",
            "timeout_seconds": 10,
            "tasks": tasks,
        }
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match configured task set"):
        QuixBugsBenchmark(config_path).discover_tasks()


@pytest.mark.integration
def test_real_quixbugs_setup_discovers_40_and_known_fix_passes():
    benchmark = QuixBugsBenchmark()
    if not benchmark.local_path.exists():
        pytest.skip("QuixBugs checkout is not set up; run python -m benchmark.setup first")
    tasks = benchmark.discover_tasks()
    assert len(tasks) == 40
    assert "node" not in tasks
    assert "breadth_first_search_test" not in tasks
    with benchmark.task_worktree("gcd") as env:
        buggy = benchmark.run_tests(env)
        assert not buggy.success
    with benchmark.task_worktree("gcd") as env:
        corrected = benchmark.local_path / "correct_python_programs" / "gcd.py"
        shutil.copyfile(corrected, env.program_path)
        fixed = benchmark.run_tests(env)
        assert fixed.success
