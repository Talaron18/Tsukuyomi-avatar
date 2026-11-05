#!/usr/bin/env python3
"""DPO preference pairs: flow.dpo. Run from the repo root (Live2D renders
need GL: xvfb-run -a, as for the other renders):

    # is the judge to be trusted? agreement on minimal-edit pairs
    xvfb-run -a python scripts/dpo_pairs.py --data data/datagen/r3 \\
        --split runs/sft/q8r3/split.json --out runs/dpo/cal \\
        --calibrate 80 --judge-backend vllm --judge-cache runs/dpo/cache

    # pairs from 4 student samples a held-out turn
    xvfb-run -a python scripts/dpo_pairs.py --data data/datagen/r3 \\
        --split runs/sft/q8r3/split.json --out runs/dpo/q8r3 \\
        --backend vllm --model q8r3 --k 4 --conc 8 \\
        --judge-backend vllm --judge-cache runs/dpo/cache --synthetic 50
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.dpo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
