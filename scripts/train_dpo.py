#!/usr/bin/env python3
"""The DPO trainer of the student: flow.train.dpo (stats, train, merge, gguf,
modelfile). Run from the repo root:

    python scripts/train_dpo.py stats --sft-run runs/sft/q8r3 \
        --data runs/dpo/q8r3/dpo.jsonl
    python scripts/train_dpo.py train --sft-run runs/sft/q8r3 \
        --data runs/dpo/q8r3/dpo.jsonl --out runs/dpo/q8r3d
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.train.dpo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
