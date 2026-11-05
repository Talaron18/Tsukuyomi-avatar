#!/usr/bin/env python3
"""Failure checkers for a student's replies: flow.checks. Run from the repo
root:

    python scripts/checks.py runs/eval_q8m/student.jsonl \\
        runs/eval_q8r2/student.jsonl --by avatar --json checks.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.checks import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
