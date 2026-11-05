#!/usr/bin/env python3
"""The style numbers of flow/style.py on a file of real streamer replies (or
chat lines), to tune the r4 teacher until it sits near real streamers. Runs
where the data lives (the CPU server); nothing is downloaded or stored.

    python scripts/style_ref.py LiveChat/Dataset/Subset/subset.json \\
        --compare data/datagen/r3 runs/eval_q8r3/recorded.jsonl \\
        --sample 820 --show 20 --json style_ref.json

PATH is read by flow.realdata (formats there); the streamer's replies are
measured when the file has pairs (--side reply), else its chat lines (--side
viewer). --compare adds columns for flow.checks inputs (a dataset folder, a
shard, an evalsft file, a live turns.jsonl), sampled alike with --sample so
that distinct-n is comparable. --show N prints N cleaned lines to read by
hand (do it before trusting a file: the keyword filter is crude).

The sampling matters: distinct-n falls as a file grows. Compare 820 against
820. The ja / zh rows are characters, the en rows words; a file of one
language has empty rows for the others.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import realdata as rd  # noqa: E402
from flow import style as sty  # noqa: E402


def _cell(v):
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--side", default="auto",
                    choices=("auto", "reply", "viewer"))
    ap.add_argument("--sample", type=int, default=820,
                    help="records measured, a seeded sample [820]; 0: all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clean", action="store_true",
                    help="measure the lines as they are in the file")
    ap.add_argument("--compare", nargs="*", default=[], metavar="PATH",
                    help="flow.checks inputs to put in further columns")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="print N of the cleaned lines to read")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    recs = rd.style_records(a.path, a.side, a.sample, a.seed,
                            not a.no_clean)
    if not recs:
        raise SystemExit(f"{a.path}: no usable lines")
    cols = [(os.path.basename(a.path), sty.style(recs))]
    if a.compare:
        from flow import checks as ck
        for p in a.compare:
            for label, rs in ck.load(p):
                rs = list(rs)
                if a.sample and len(rs) > a.sample:
                    rs = random.Random(f"{a.seed}:cmp").sample(rs, a.sample)
                cols.append((label, sty.style(rs)))
    rows = [dict(sty.rows_of(c)) for _l, c in cols]
    labs = [lab for lab, _ in sty.rows_of(cols[0][1])]
    w0 = max(len(x) for x in labs) + 2
    wc = max([14] + [len(l) + 2 for l, _ in cols])
    print(f"{'':{w0}s}" + "".join(f"{l:>{wc}s}" for l, _ in cols))
    for lab in labs:
        print(f"{lab:{w0}s}" + "".join(f"{_cell(r.get(lab)):>{wc}s}"
                                      for r in rows))
    if a.show:
        print()
        for r in random.Random(a.seed).sample(recs, min(a.show, len(recs))):
            print(("viewer: " + r["user"] + "\n  " if r["user"] else "")
                  + r["spoken"])
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({l: c for l, c in cols}, fh, ensure_ascii=False,
                      indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
