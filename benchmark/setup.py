from __future__ import annotations

from benchmark.quixbugs import QuixBugsBenchmark


def main() -> None:
    benchmark = QuixBugsBenchmark()
    benchmark.setup()
    print(f"QuixBugs ready at {benchmark.local_path}")
    print(f"Benchmark commit: {benchmark.benchmark_commit()}")


if __name__ == "__main__":
    main()

