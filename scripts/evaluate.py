#!/usr/bin/env python3
"""Script to run the automated AVY evaluation benchmark."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avy.evaluation.harness import EvaluationHarness


def main() -> None:
    print("Running AVY Automated Evaluation Benchmark...")
    harness = EvaluationHarness()
    report = harness.run_benchmark()
    summary = harness.format_summary_table(report)
    print("\n" + summary + "\n")


if __name__ == "__main__":
    main()
