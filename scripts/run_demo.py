#!/usr/bin/env python3
"""Script to run the interactive CLI demonstration across all 6 Theme 04 scenarios."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from avy.cli import cmd_demo
from avy.config import AVYConfig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AVY interactive CLI scenarios")
    parser.add_argument("--scenario", default=None, help="Filter by scenario ID (e.g. A, B, C, D, E, F)")
    parser.add_argument("--provider", default=None, help="LLM Provider override (e.g. mock, ollama)")
    args = parser.parse_args()

    overrides = {}
    if args.provider:
        overrides["provider"] = args.provider
    config = AVYConfig.load(**overrides)
    cmd_demo(args, config)
