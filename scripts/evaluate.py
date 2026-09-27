#!/usr/bin/env python3
"""Script to run the automated AVY evaluation benchmark."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avy.config import AVYConfig
from avy.evaluation.dataset import get_standard_eval_dataset
from avy.evaluation.harness import EvaluationHarness
from avy.providers.registry import get_provider
from avy.streaming.engine import StreamingLiveRAGEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="AVY Automated Evaluation Benchmark")
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "ollama", "openai"],
        help="LLM Provider override for evaluation (default: mock for fast, deterministic evaluation)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export benchmark results as JSON",
    )
    args = parser.parse_args()

    print(f"Running AVY Automated Evaluation Benchmark (Provider: {args.provider})...")
    config = AVYConfig.load(provider=args.provider)
    provider = get_provider(args.provider, config=config)
    engine = StreamingLiveRAGEngine(config=config, provider=provider)
    harness = EvaluationHarness(engine=engine)

    dataset = get_standard_eval_dataset()
    print(f"Evaluating {len(dataset)} standard Theme 04 benchmark test scenarios...\n")

    report = harness.run_benchmark(dataset=dataset)
    summary = harness.format_summary_table(report)
    print(summary + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert TestResult objects to dicts
        serializable_report = dict(report)
        serializable_report["results"] = [
            r.__dict__ if hasattr(r, "__dict__") else r for r in report["results"]
        ]
        out_path.write_text(json.dumps(serializable_report, indent=2), encoding="utf-8")
        print(f"Report saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
