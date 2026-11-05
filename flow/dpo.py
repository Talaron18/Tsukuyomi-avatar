#!/usr/bin/env python3
"""
DPO preference pairs for the student, from three signals: the text checkers
(flow.checks), the vision judge (flow.judge) and minimal edits of the
teacher's own replies. Run through scripts/dpo_pairs.py; the trainer is
flow.train.dpo, which reads the jsonl (prompt / chosen / rejected as chat
messages, TRL's conversational format) and dpo.meta.jsonl.

Prompts. For each turn the messages are the SFT export's for that turn
(flow.dataexport.sft_examples, per="turn": the system prompt of the avatar
and persona -- or --student-prompt short --, the history as the record has
it, the user turn with its tail lines), with the final assistant message
cut; "weight" keys of the history are dropped. The same turns, replies and
manners as `dataexport sft` (dataexport.prepare), so a turn the export drops
is not a pair either. The teacher's reply for the turn is the export's
target (corrections and manners applied).

Candidates. (i) K student samples a turn through a director backend (ollama,
vllm ...), the student's sampling, a seed per sample; or (ii) a jsonl of
{"id", "reply"} lines (an evalsft student.jsonl works: "reply_raw" is read
too). --teacher-chosen adds the teacher's reply to each turn's pool.

Score of a candidate (higher is better), text and vision together:

    score = -25 * H  - 1 * S
            - 5 * (2 - request_done) - 5 * (2 - visible)

H: hard text failures, each failing check once: look_sign (the wrong side or
no turn), body_request_obeyed / wave_obeyed (a skipped request), language,
template_repeat. S: soft: opening (the q8r2 template), beats < sentences,
quiet_says_quiet, frame_before_wave. request_done and visible are the
judge's 0-2 (NA counts as 2: no penalty). --judge-dims (default
request_done,visible) picks the judge fields that enter the score; adding
expression_fit or natural gives the older terms "+ 1 * expression_fit
+ 0.5 * (natural - 1)" (missing: 1 and 3). Left out they are still recorded
in the meta (and --min-natural reads natural), they just do not move the
score: on the calibration pairs they were mostly ties. For a look request
(left / right) the text checker decides the side: look_sign is a hard
failure, and the judge's request_done is not scored or compared with it
(small head turns are hard to see in a tile); its visible still is. The
weights make each level outweigh everything below it: one hard failure (25)
is more than the judge's primary scores can move (5 a step, 20 over both),
and one primary step (5) is more than the optional secondary range (4); soft
failures (1) only break near ties. A candidate the judge could not see (a
VRM, a model that is not here, --no-judge) is scored on text alone.

Contradictions. The text and the judge disagree on a request when
(a) a request check failed and the judge says request_done = 2, or
(b) every request check applicable passed and the judge says request_done = 0
    while it also sees a body move that is visible (visible 1 or 2).
Such a candidate is left out of the turn (neither chosen nor rejected: one
of the two signals is wrong and we cannot tell which). When the checks pass
and the judge says visible = 0, or sees no move at all (null) with
request_done = 0 (a hand out of the picture in bust: the text sees a hand
marker, the viewer no hand), visibility is the reason: the
candidate may be rejected, never chosen. A check that fails with the judge
saying request_done < 2, or that passes with the judge saying 2, is
agreement.

Pairs. chosen = the best candidate with no hard failure and no visibility
flag, rejected = the worst of the others; kept when chosen - rejected >=
--min-gap (5: one primary step, so a pair rests on a hard failure or a
request / visibility difference, never on expression and naturalness alone;
lower it to learn from those), the replies differ, and (--min-natural) the
chosen one looks natural. Dropped pairs are counted by reason.

Minimal-edit pairs (no judge): the teacher's reply as chosen and a copy
that breaks one thing as rejected, kind in the meta: look_flip (the first
turning marker's head.yaw and gaze.x change sign; on look requests the
teacher got right), hand_drop (every hand / move / gesture marker or hand
target removed; on hand and wave requests it fulfilled), template_open (the
first beat marker becomes the old q8r2 opening: brows 0.7, pitch 5, yaw -5,
dur 0.2, hold 0.4), beat_dup (a beat marker repeated before the next beat,
or after the next sentence), wave_frame (a `[frame full]` right before the
first wave move of a reply that has none before it: the student does that
32% of the time, the teacher 4%; frame_before_wave sees it), opening_repeat
(the first beat becomes the student's most common opening, from
--opening-from a student.jsonl, flow.checks' opening stats; where the
teacher's opening differs), look_at_me (on "look at me / at the camera"
lines: chosen keeps head.yaw within 3 degrees and gaze.x within 0.15, the
teacher's reply or edited to it; rejected turns those markers 12-20 degrees
sideways with the matching gaze.x; checker look_at_me), correction (the
user corrects a word: chosen says the right form Y, rejected has Y swapped
back to X in its speech; real correction lines, else a synthetic injected
one: the prompt's last reply gets Y misspelled and the user line becomes the
correction; checker correction_obeyed), pose_copy (rejected pastes the pose
markers of the last earlier reply onto this reply; checker pose_copy),
glance_at_mention (on a line that puts something near her without asking
her to look, flow.checks.mentioned_place: chosen opens with a glance beat
toward it, added when the teacher's reply has none in its first two beats;
rejected is the teacher's reply without it, or, when the teacher glances,
that reply with the glance's sign reversed; speech identical; checker
glance_at_mention). The
spoken text is untouched except by correction. `--kinds` picks
the kinds, `--kind-n look_flip=100,wave_frame=40` a count per kind, and the
meta of each pair has the language of the user line ("lang": ja / zh / en),
counted in the summary, so look_flip can be weighted toward Japanese lines
in the trainer's mixing (`--data a.jsonl:1,b.jsonl:2`).

Calibration (--calibrate N): N such pairs, balanced over the kinds, are
judged on both sides; per kind the judge's vision score and the kind's own
dimension (look_flip, hand_drop: request_done; template_open, beat_dup:
natural) give win (the intact reply scores higher), tie, loss. Trust the
judge for pairs only where it wins clearly.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import threading
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    from . import checks as ck
    from . import dataexport as dx
    from . import dataset as ds
    from . import director as dr
    from . import judge as jg
    from . import pose as pz
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from flow import checks as ck
    from flow import dataexport as dx
    from flow import dataset as ds
    from flow import director as dr
    from flow import judge as jg
    from flow import pose as pz

HARD = ("look_sign", "look_at_me", "body_request_obeyed", "wave_obeyed",
        "correction_obeyed", "language", "template_repeat", "pose_copy")
SOFT = ("opening", "beats", "quiet_says_quiet", "frame_before_wave",
        "glance_at_mention")
REQUEST = ("look_sign", "look_at_me", "body_request_obeyed", "wave_obeyed")
W_HARD, W_SOFT, W_PRIMARY, W_FIT, W_NAT = 25.0, 1.0, 5.0, 1.0, 0.5
MIN_GAP = 5.0
# The judge fields that enter the pair score (--judge-dims). The others
# (expression_fit, natural) are recorded in the meta and read by
# --min-natural but add nothing to the score: on the calibration pairs they
# were noise (module docstring, Calibration).
DIMS = ("request_done", "visible")
JUDGE_DIMS = DIMS
# The kinds the judge is calibrated on, and the ones added for q8r3's
# failures (wave_frame, opening_repeat: text checkers decide them; the judge
# does not see a framing or an opening, so they are not calibrated).
CAL_KINDS = ("look_flip", "hand_drop", "template_open", "beat_dup")
KINDS = CAL_KINDS + ("wave_frame", "opening_repeat", "look_at_me",
                     "correction", "pose_copy", "glance_at_mention")
KIND_DIM = {"look_flip": "request_done", "hand_drop": "request_done",
            "template_open": "natural", "beat_dup": "natural",
            "wave_frame": "natural", "opening_repeat": "natural",
            "look_at_me": "request_done", "correction": "natural",
            "pose_copy": "natural", "glance_at_mention": "natural"}
# look_at_me: the rejected side turns the head this far (degrees, by a hash
# of the reply) and the eyes with it (gaze.x per degree, pose.py's units).
SIDE_YAW = (12, 20)
SIDE_GAZE_PER_DEG = 0.03
# The q8r2 opening, as the old prompt's dense example had it.
TEMPLATE = {"brow.L.lift": 0.7, "brow.R.lift": 0.7, "head.pitch": 5,
            "head.yaw": -5, "dur": 0.2, "hold": 0.4}


def _quiet(_s):
    pass


# ---------------------------------------------------------------- the turns

def split_sessions(path, side="eval") -> set:
    """The sessions of one side of a flow.train.sft split.json."""
    with open(path, encoding="utf-8") as fh:
        return set(json.load(fh)[f"{side}_sessions"])


def clean_messages(msgs) -> list:
    """The export's messages without the history's "weight" keys (a trainer
    and a chat template read role and content)."""
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def load_turns(data, sessions=None, avatars=None, limit=None,
               student="teacher", log=_quiet, **prep) -> list:
    """[{id, rec, prompt, teacher}] for the turns the SFT export keeps
    (per turn) of these sessions (None: all) and avatars: `prompt` the
    messages up to the user turn, `teacher` the export's reply for it.
    `prep`: dataexport.prepare's options (manners, language filter,
    keep_lite ...), as `dataexport sft` takes them: a live2d-lite avatar's
    turns are not in the export, so not here either, unless keep_lite."""
    recs = ds.iter_records(data)
    plan = dx.prepare(data, recs, **prep)
    want = [r for r in recs if (sessions is None or r["session"] in sessions)
            and (not avatars or r["avatar"]["name"] in avatars)]
    exs = dx.sft_examples(data, want, plan["notes"], "turn", True,
                          {**plan["fix"], **plan["mfix"]}, None, student)
    by_id = {r["id"]: r for r in want}
    out = []
    for ex in exs:
        rid = ex["meta"]["id"]
        msgs = ex["messages"]
        if msgs[-1]["role"] != "assistant":
            continue
        out.append({"id": rid, "rec": by_id[rid],
                    "prompt": clean_messages(msgs[:-1]),
                    "teacher": msgs[-1]["content"]})
    out.sort(key=lambda t: (t["rec"]["session"], t["rec"]["turn"]))
    if limit:
        out = out[:limit]
    log(f"dpo: {len(out)} turns of {len(want)} records (the export drops "
        f"{len(want) - len(out)})")
    return out


# ------------------------------------------------------------- the candidates

def read_candidates(paths) -> dict:
    """{turn id: [(source, reply)]} from jsonl files of {"id", "reply"}
    (also "reply_raw", as evalsft writes; "name" / "source" labels)."""
    out = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                reply = r.get("reply") if "reply" in r else r.get("reply_raw")
                if reply is None or "id" not in r:
                    continue
                out[r["id"]].append((str(r.get("source") or r.get("name")
                                         or os.path.basename(path)), reply))
    return dict(out)


def sample(turns, k, args, conc=1, seed=0, log=_quiet) -> dict:
    """{turn id: [(source, reply)]}: k replies a turn from the backend of
    `args` (director's: --backend ollama | vllm ..., --model). Each sample
    has its own seed (the Ollama backend sends a fixed one: the same seed
    would give k copies) and its own backend (usage is per instance)."""
    todo = [(t, j) for t in turns for j in range(k)]
    out = defaultdict(dict)
    lock = threading.Lock()
    done = [0]

    def one(item):
        t, j = item
        a = argparse.Namespace(**vars(args))
        a.seed = int(seed) + 7919 * j + (zlib.crc32(t["id"].encode("utf-8")) & 0xFFFF)
        reply = None
        for attempt in range(3):
            try:
                be = dr.make_backend(a, log=_quiet)
                reply = "".join(be.stream(t["prompt"][0]["content"],
                                          t["prompt"][1:]))
                break
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 - a network or server error
                log(f"dpo: {t['id']} sample {j}: {exc}")
        with lock:
            if reply is not None:
                out[t["id"]][j] = reply
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"dpo: sampled {done[0]}/{len(todo)}")

    if conc <= 1:
        for item in todo:
            one(item)
    else:
        with ThreadPoolExecutor(max_workers=conc) as pool:
            list(pool.map(one, todo))
    return {i: [(f"student#{j}", r) for j, r in sorted(d.items())]
            for i, d in out.items()}


# ------------------------------------------------------------- text checks

def check_replies(turn, replies, bodies) -> dict:
    """{source: {verdicts {check: pass/fail/na}, why}} for this turn's
    replies [(source, raw)]: each as its own session whose previous turn is
    the history's last reply (template_repeat), the bare user line, the
    turn's framing and language mix."""
    r = turn["rec"]
    avatar = r["avatar"]["name"]
    fr_ = r.get("framing")
    start = fr_.get("start") if isinstance(fr_, dict) else None
    mix = (r.get("lang") or {}).get("mix")
    hist = r.get("history") or []
    recs, ids = [], {}
    for n, (src, raw) in enumerate(replies):
        sid = f"{r['session']}#{n}"
        for i, h in enumerate(hist[-ck.COPY_PREV:]):
            back = len(hist[-ck.COPY_PREV:]) - i
            recs.append(ck.make_rec(h.get("reply", ""),
                                    h.get("user", "") if i == len(
                                        hist[-ck.COPY_PREV:]) - 1 else "",
                                    avatar, sid, r["turn"] - back,
                                    rid=f"{sid}/prev{i}"))
        rec = ck.make_rec(raw, r["user"], avatar, sid, r["turn"], rid=sid,
                          quiet=r.get("quiet"), start=start,
                          topic=r.get("topic"), mix=mix)
        recs.append(rec)
        ids[src, n] = sid
    out = {}
    verdicts = {}
    for name in HARD + SOFT:
        v, _extra, why = ck.CHECKS[name](recs, bodies)
        verdicts[name] = (v, why)
    for n, (src, _raw) in enumerate(replies):
        sid = ids[src, n]
        out[n] = {"verdicts": {c: verdicts[c][0].get(sid, ck.NA)
                               for c in HARD + SOFT},
                  "why": {c: verdicts[c][1][sid] for c in HARD + SOFT
                          if sid in verdicts[c][1]}}
    return out


def failures(verdicts):
    """(hard failing checks, soft failing checks)."""
    return ([c for c in HARD if verdicts.get(c) == ck.FAIL],
            [c for c in SOFT if verdicts.get(c) == ck.FAIL])


# ------------------------------------------------------------ the score

def vision_score(j, look=False, dims=None) -> float:
    """The judge's part of the score (module docstring); 0 without one.
    Only `dims` (JUDGE_DIMS) count. look: the viewer asked for a look
    left / right, whose side the text checker decides (look_sign): the
    judge's request_done is not scored then."""
    if not j:
        return 0.0
    dims = JUDGE_DIMS if dims is None else dims
    out = 0.0
    if "request_done" in dims and not look:
        rd = 2 if j.get("request_done") is None else j["request_done"]
        out -= W_PRIMARY * (2 - rd)
    if "visible" in dims:
        vi = 2 if j.get("visible") is None else j["visible"]
        out -= W_PRIMARY * (2 - vi)
    if "expression_fit" in dims:
        out += W_FIT * (1 if j.get("expression_fit") is None
                        else j["expression_fit"])
    if "natural" in dims:
        out += W_NAT * ((3 if j.get("natural") is None
                         else j["natural"]) - 1)
    return out


def combined(verdicts, j, look=False) -> float:
    hard, soft = failures(verdicts)
    return -W_HARD * len(hard) - W_SOFT * len(soft) + vision_score(j, look)


def conflict(verdicts, j, look=False):
    """(None | "visibility" | "contradiction", reason): how the text and the
    judge stand on the request (module docstring). look: a look request,
    where the text checker decides and the judge's request_done is not
    read (only whether it saw the picture)."""
    if not j:
        return None, ""
    rv = [verdicts.get(c) for c in REQUEST]
    t_fail = ck.FAIL in rv
    t_ok = ck.PASS in rv and not t_fail
    rd, vi = j.get("request_done"), j.get("visible")
    if look:
        rd = None
    if t_fail and rd == 2:
        return "contradiction", "text: request failed; judge: done"
    if t_ok and rd == 0:
        # A hand cropped out of a bust still may be answered "not visible"
        # or "no body move" (null): both are the picture, not the markers.
        if vi in (0, None):
            return "visibility", "text: done; judge: not visible"
        return "contradiction", "text: done; judge: not done"
    if t_ok and vi == 0:
        return "visibility", "text: done; judge: not visible"
    return None, ""


# --------------------------------------------------------- the candidates

class Context:
    """What evaluating a turn needs: the bodies (checks), the avatars
    (renderers), the judge or None, the render options; `stopped` once the
    judge's budget ran out."""

    def __init__(self, bodies, avatars=None, judge=None, tiles=jg.MAX_TILES,
                 tile=jg.TILE, follow=True, conc=1, sheets=None, log=_quiet):
        self.bodies, self.avatars, self.judge = bodies, avatars, judge
        self.tiles, self.tile, self.follow, self.conc = (tiles, tile, follow,
                                                         conc)
        self.sheets, self.log = sheets, log
        self.stopped = False
        self.na = Counter()


def judge_replies(turn, replies, ctx) -> dict:
    """{index: {"scores", "raw", "cached"} | {"na": why}} for replies
    [(source, raw)]: the sheets are rendered one after the other (GL stays on
    one thread), the judge calls run `conc` at a time."""
    n = len(replies)
    if ctx.judge is None or ctx.avatars is None:
        return {i: {"na": "no judge"} for i in range(n)}
    r = turn["rec"]
    name = r["avatar"]["name"]
    bm, rend, why = ctx.avatars.get(name)
    if rend is None:
        ctx.na[why] += 1
        return {i: {"na": why} for i in range(n)}
    out, sheets, seen = {}, {}, {}
    for i, (_src, raw) in enumerate(replies):
        if raw in seen:
            sheets[i] = sheets[seen[raw]]
            continue
        seen[raw] = i
        try:
            sheets[i] = jg.render_sheet(
                bm, rend, raw, jg.home_of(r), jg.now_targets(
                    r["user_content"]), r.get("seed", 0), ctx.tiles,
                ctx.tile, ctx.follow, name)
        except Exception as exc:  # noqa: BLE001 - a render failure is NA
            ctx.log(f"dpo: {turn['id']}: render failed: {exc}")
            ctx.na["render failed"] += 1
            sheets[i] = None
        if ctx.sheets and sheets[i] is not None:
            os.makedirs(ctx.sheets, exist_ok=True)
            safe = f"{turn['id'].replace('/', '_')}-{i}.png"
            with open(os.path.join(ctx.sheets, safe), "wb") as fh:
                fh.write(sheets[i].png)
    budget = ctx.bodies.budget(name)

    def one(i):
        sh = sheets[i]
        if sh is None:
            return i, {"na": "render failed"}
        try:
            got = ctx.judge.score(sh, r["user_content"],
                                  dx.spoken_of(replies[i][1]), budget, name)
        except jg.BudgetExceeded:
            ctx.stopped = True
            return i, {"na": "judge budget"}
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad call is NA
            ctx.log(f"dpo: {turn['id']}: judge failed: {exc}")
            return i, {"na": f"judge failed: {exc}"}
        if got["scores"] is None:
            return i, {"na": "unparsed judgment", "raw": got["raw"]}
        return i, got

    todo = sorted(set(seen.values()))
    if ctx.conc > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=ctx.conc) as pool:
            res = dict(pool.map(one, todo))
    else:
        res = dict(one(i) for i in todo)
    for i in range(n):
        out[i] = res[i] if i in res else res[seen[replies[i][1]]]
    return out


def evaluate_turn(turn, replies, ctx) -> list:
    """The scored candidates of a turn: dicts {turn, src, reply, verdicts,
    why, hard, soft, judge (the scores or None), judge_raw, na, conflict,
    reason, score}. Identical replies are one candidate (sources joined)."""
    uniq, order = {}, []
    for src, raw in replies:
        if raw in uniq:
            uniq[raw].append(src)
        else:
            uniq[raw] = [src]
            order.append(raw)
    reps = [("+".join(uniq[raw]), raw) for raw in order]
    texts = check_replies(turn, reps, ctx.bodies)
    judged = judge_replies(turn, reps, ctx)
    look = bool(ck.asked_side(turn["rec"]["user"])
                or ck.asked_at_me(turn["rec"]["user"]))
    out = []
    for i, (src, raw) in enumerate(reps):
        v = texts[i]["verdicts"]
        hard, soft = failures(v)
        jr = judged[i]
        j = jr.get("scores")
        kind, reason = conflict(v, j, look)
        out.append({"turn": turn["id"], "src": src, "reply": raw,
                    "verdicts": v, "why": texts[i]["why"], "hard": hard,
                    "soft": soft, "judge": j, "judge_raw": jr.get("raw"),
                    "judge_cached": bool(jr.get("cached")),
                    "na": jr.get("na"), "conflict": kind, "reason": reason,
                    "chosen_only": "teacher" in uniq[raw],
                    "score": combined(v, j, look)})
    return out


# -------------------------------------------------------------- the choice

def choose(cands, min_gap=MIN_GAP, min_natural=0, allow_hard_chosen=False):
    """(chosen, rejected, why): the pair of a turn's candidates, or
    (None, None, reason): no_candidates, contradictions, no_clean_chosen,
    no_rejected, identical, gap, natural. A candidate with chosen_only (the
    teacher's reply) is never the rejected one."""
    if len(cands) < 2:
        return None, None, "no_candidates"
    pool = [c for c in cands if c["conflict"] != "contradiction"]
    if len(pool) < 2:
        return None, None, "contradictions"
    ok = [c for c in pool if c["conflict"] is None
          and (allow_hard_chosen or not c["hard"])]
    if not ok:
        return None, None, "no_clean_chosen"
    best = max(ok, key=lambda c: c["score"])
    # The teacher's reply is the SFT target: a chosen candidate only.
    rest = [c for c in pool if c is not best and not c.get("chosen_only")]
    if not rest:
        return None, None, "no_rejected"
    worst = min(rest, key=lambda c: c["score"])
    if best["reply"] == worst["reply"]:
        return None, None, "identical"
    if best["score"] - worst["score"] < min_gap:
        return None, None, "gap"
    nat = (best["judge"] or {}).get("natural")
    if min_natural and nat is not None and nat < min_natural:
        return None, None, "natural"
    return best, worst, None


def pair_row(turn, chosen, rejected) -> dict:
    return {"prompt": copy.deepcopy(turn["prompt"]),
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]}


def cand_meta(c) -> dict:
    return {k: c[k] for k in ("src", "score", "hard", "soft", "judge",
                              "conflict", "na") if k in c}


# ---------------------------------------------------------- minimal edits

def _json_part(text, a, b):
    i, j = text.find("{", a, b), text.rfind("}", a, b)
    return (i, j + 1) if 0 <= i < j else None


def _cut(text, a, b):
    """text without text[a:b], a marker, and the space that would double
    (the parser drops the space after a marker that follows one; two in a
    row inside the text would stay in the spoken words)."""
    if b < len(text) and text[b] == " " and (a == 0 or text[a - 1] in " \n"):
        b += 1
    return text[:a] + text[b:]


def _marker_edit(raw, span, fn):
    """raw with the pose marker at span (marker_spans') rewritten: fn gets
    the marker's JSON as an ordered dict and returns the new dict, or None
    to delete the marker. None when the JSON is not strict."""
    a, b, _t = span
    part = _json_part(raw, a, b)
    if part is None:
        return False
    try:
        data = json.loads(raw[part[0]:part[1]])
    except ValueError:
        return False
    new = fn(dict(data))
    if new is None:
        return _cut(raw, a, b)
    return raw[:part[0]] + json.dumps(new, ensure_ascii=False) + raw[part[1]:]


def _spans(raw):
    return dx.marker_spans(raw)


def _rec(raw, user="", avatar="x"):
    return ck.make_rec(raw, user, avatar, "edit", 1, rid="edit")


def flip_look(raw, user):
    """raw with the first turning marker's head.yaw and gaze.x flipped, for
    a look request the reply gets right; None when it does not apply."""
    side = ck.asked_side(user)
    if side is None:
        return None
    want = -1 if side == "R" else 1
    if ck.first_turn(_rec(raw)) != want:
        return None
    for sp in _spans(raw):
        t = sp[2]
        if t["tag"] != pz.POSE:
            continue
        tg = t["arg"]["targets"]
        yaw, gx = ck._num(tg.get("head.yaw")), ck._num(tg.get("gaze.x"))
        if not ((yaw is not None and abs(yaw) >= ck.YAW_MIN)
                or (gx is not None and abs(gx) >= ck.GAZE_MIN)):
            continue

        def flip(d):
            for k in ("head.yaw", "gaze.x"):
                if isinstance(d.get(k), (int, float)) \
                        and not isinstance(d[k], bool):
                    d[k] = -d[k]
            return d
        new = _marker_edit(raw, sp, flip)
        if new is False or new is None:
            return None
        if ck.first_turn(_rec(new)) == -want:
            return new
        return None
    return None


def drop_hand(raw, user, has_hands=True):
    """raw without the hand: moves and gestures removed, hand targets taken
    out of poses (a pose left with nothing goes), for a hand or wave
    request the reply fulfils; None when it does not apply."""
    if not has_hands or not (ck.asked_hands(user) or ck.asked_wave(user)):
        return None
    if not ck.has_hand(_rec(raw)):
        return None
    out = raw
    for sp in reversed(_spans(raw)):
        t = sp[2]
        if t["tag"] in ("move", "gesture"):
            out = _cut(out, sp[0], sp[1])
        elif t["tag"] == pz.POSE and any(dx.is_hand(k)
                                         for k in t["arg"]["targets"]):
            def strip(d):
                d = {k: v for k, v in d.items() if k in ("dur", "hold")
                     or not dx.is_hand(k)}
                return d if any(k not in ("dur", "hold") for k in d) else None
            # `out` ahead of this span is unchanged: spans go back to front.
            new = _marker_edit(out, sp, strip)
            if new is False:
                return None
            out = new
    if ck.has_hand(_rec(out)) or out == raw:
        return None
    return out


def template_open(raw, user, offered=None):
    """raw with its first pose marker, a beat, as the q8r2 opening; None
    when the reply has no such marker, already opens so, the beat is the
    look the user asked for, or the body lacks the template's variables
    (`offered`: the names it has)."""
    if offered is not None and not set(k for k in TEMPLATE
                                       if k not in ("dur", "hold")) \
            <= set(offered):
        return None
    for sp in _spans(raw):
        t = sp[2]
        if t["tag"] != pz.POSE:
            continue
        tg = t["arg"]["targets"]
        if not dx.is_beat(tg.keys()) or ck.is_q8r2_opening(_rec(raw)):
            return None
        if ck.asked_side(user) and ck._num(tg.get("head.yaw")) is not None \
                and abs(ck._num(tg["head.yaw"])) >= ck.YAW_MIN:
            return None
        if ck.asked_at_me(user):         # its yaw -5 would turn her away
            return None
        new = _marker_edit(raw, sp, lambda d: dict(TEMPLATE))
        if new is False or new is None or not ck.is_q8r2_opening(_rec(new)):
            return None
        return new
    return None


def beat_dup(raw):
    """raw with its first beat marker repeated: before the next beat
    marker, else after the end of the next sentence, else right after it;
    None without a beat."""
    spans = _spans(raw)
    beats = [s for s in spans if s[2]["tag"] == pz.POSE
             and dx.is_beat(s[2]["arg"]["targets"].keys())]
    if not beats:
        return None
    a, b, _t = beats[0]
    marker = raw[a:b]
    if len(beats) > 1:
        at = beats[1][0]
    else:
        m = re.search(r"[.!?。！？]+\s*", raw[b:])
        at = b + m.end() if m else b
    pre = "" if at == b or raw[at - 1:at].isspace() else " "
    return raw[:at] + pre + marker + " " + raw[at:]


def wave_frame(raw):
    """raw with `[frame full]` right before its first wave move, when no
    full frame comes before it (the checker's frame_before_wave fails then);
    None without a wave, or when a full frame already precedes it."""
    spans = _spans(raw)
    i = ck._first_index([sp[2] for sp in spans], lambda t: t["tag"] == "move"
                        and (t["arg"] or {}).get("name") == "wave")
    if i is None or any(ck._is_full(sp[2]) for sp in spans[:i]):
        return None
    a = spans[i][0]
    pre = "" if a == 0 or raw[a - 1].isspace() else " "
    new = raw[:a] + pre + "[frame full] " + raw[a:]
    if ck.check_frame_before_wave([_rec(raw)], None)[0]["edit"] != ck.PASS \
            or ck.check_frame_before_wave([_rec(new)], None)[0]["edit"] \
            != ck.FAIL:
        return None
    return new


def _centred(d) -> dict:
    """d with head.yaw and gaze.x brought to 0 where they are set but off
    centre (more than AT_ME_YAW / AT_ME_GAZE)."""
    d = dict(d)
    for k, lim in (("head.yaw", ck.AT_ME_YAW), ("gaze.x", ck.AT_ME_GAZE)):
        v = ck._num(d.get(k))
        if v is not None and abs(v) > lim:
            d[k] = 0
    return d


def _is_centred(raw) -> bool:
    """Every head.yaw / gaze.x the poses set is within AT_ME_YAW /
    AT_ME_GAZE, and at least one is set."""
    seen = False
    for sp in _spans(raw):
        t = sp[2]
        if t["tag"] != pz.POSE:
            continue
        for k, lim in (("head.yaw", ck.AT_ME_YAW), ("gaze.x", ck.AT_ME_GAZE)):
            v = ck._num(t["arg"]["targets"].get(k))
            if v is not None:
                seen = True
                if abs(v) > lim:
                    return False
    return seen


def look_at_me(raw, user, offered=None):
    """(chosen, rejected) for a "look at me / at the camera" line, or None.
    chosen: a reply whose pose markers keep head.yaw and gaze.x near 0 (the
    teacher's own reply when it does; else the reply with the off-centre
    values brought to 0, or head.yaw 0 / gaze.x 0 set on its first pose
    marker when it sets none). rejected: that reply with the first marker
    that sets head.yaw / gaze.x turned sideways, 12-20 degrees and the
    matching gaze.x, the side from a hash of the reply. Speech identical.
    `offered`: the variables the body has (head.yaw and gaze.x both needed
    to add them)."""
    if not ck.asked_at_me(user):
        return None
    spans = _spans(raw)
    if not any(sp[2]["tag"] == pz.POSE for sp in spans):
        return None
    chosen = raw
    if not _is_centred(raw):
        # Centre what is off; where nothing is set, add 0s to the first pose.
        out = raw
        for sp in reversed(spans):
            if sp[2]["tag"] != pz.POSE:
                continue
            tg = sp[2]["arg"]["targets"]
            if any(ck._num(tg.get(k)) is not None
                   for k in ("head.yaw", "gaze.x")):
                new = _marker_edit(out, sp, _centred)
                if new is False or new is None:
                    return None
                out = new
        if not _is_centred(out):
            if offered is not None and not {"head.yaw", "gaze.x"} <= set(
                    offered):
                return None
            first = next(sp for sp in _spans(out)
                         if sp[2]["tag"] == pz.POSE)
            new = _marker_edit(out, first, lambda d: {
                **d, "head.yaw": 0, "gaze.x": 0})
            if new is False or new is None:
                return None
            out = new
        chosen = out
    if not _is_centred(chosen):
        return None
    h = zlib.crc32(raw.encode("utf-8"))
    side = 1 if h % 2 else -1
    deg = SIDE_YAW[0] + (h >> 1) % (SIDE_YAW[1] - SIDE_YAW[0] + 1)
    gaze = round(deg * SIDE_GAZE_PER_DEG, 2)

    def turn(d):
        d["head.yaw"] = side * deg
        d["gaze.x"] = side * gaze
        return d
    new = chosen
    # Every marker that sets head.yaw / gaze.x turns, so the reply ends on
    # the side too (the checker reads where it ends up).
    for sp in reversed(_spans(chosen)):
        tg = sp[2]["arg"]["targets"] if sp[2]["tag"] == pz.POSE else {}
        if ck._num(tg.get("head.yaw")) is None \
                and ck._num(tg.get("gaze.x")) is None:
            continue
        new = _marker_edit(new, sp, turn)
        if new is False or new is None:
            return None
    if new == chosen:
        return None
    # The reply must now end sideways (the checker's own view), and the
    # chosen one centred.
    yaw, gx = ck.final_look(_rec(new))
    if yaw is None or abs(yaw) < ck.YAW_MIN or gx is None \
            or abs(gx) < ck.GAZE_MIN:
        return None
    return chosen, new


# ------------------------------------------------------------ corrections

def _outside_markers(raw, fn):
    """raw with fn applied to the text between markers only."""
    out, pos = [], 0
    for a, b, _t in _spans(raw):
        out.append(fn(raw[pos:a]))
        out.append(raw[a:b])
        pos = b
    out.append(fn(raw[pos:]))
    return "".join(out)


def _swap_word(raw, a, b):
    """raw with every a in its speech replaced by b (markers untouched)."""
    return _outside_markers(raw, lambda t: t.replace(a, b))


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{3,}|[\u30a0-\u30ff\u4e00-\u9fff"
                   r"ー]{3,}|[\u3040-\u309f]{5,}")


def _words_of(text):
    """Words worth correcting: Latin words of 4+ letters, katakana / kanji
    runs of 3+ (names, loanwords), long hiragana runs."""
    return set(_WORD.findall(text or ""))


def misspell(w, h=0):
    """A misspelling of w: two neighbouring characters swapped (inside the
    word when it has four or more), the place from the hash h; None when no
    swap changes it."""
    n = len(w)
    if n < 3:
        return None
    lo, hi = (1, n - 2) if n >= 4 else (n - 2, n - 2)
    for k in range(hi - lo + 1):
        i = lo + (h + k) % (hi - lo + 1)
        if w[i] != w[i + 1]:
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return None


CORRECTION_LINES = {
    "ja": ("{x}じゃなくて{y}だよ！", "{y}だよ、{x}じゃない", "違う、{y}だってば"),
    "zh": ("不是{x}，是{y}！", "是{y}不是{x}", "{y}才对，不是{x}"),
    "en": ("It's \"{y}\", not \"{x}\"!", "Not {x}, {y}.", "No, {y}, not {x}.")}


def correction(raw, user, prev=None):
    """(chosen, rejected) for a line that corrects a word of hers: chosen
    the teacher's reply, which says the right form Y and not the wrong X;
    rejected the same reply with Y swapped back to X in its speech (markers
    untouched). `prev`: her last reply's spoken text, for corrections that
    name only the right form. None when the line corrects nothing or the
    reply does not say Y."""
    sp = dx.spoken_of(raw)
    for x, y in ck.corrections(user, prev):
        if not y or y not in sp or ck.says_wrong(sp, x, y):
            continue
        new = _swap_word(raw, y, x)
        if new != raw and ck.says_wrong(dx.spoken_of(new), x, y):
            return raw, new
    return None


def inject_correction(turn, raw, user=None):
    """A synthetic correction turn from a real one: (turn', chosen,
    rejected, info) or None. The turn's last assistant message in the prompt
    gets a word W she used (a word that is also in the teacher's reply to
    this turn) misspelled to X, the user line becomes a correction ("X
    じゃなくて W だよ", "It's W, not X", "不是X，是W") in the language of the
    original line, and the teacher's reply (it says W) is the chosen side,
    with W swapped to X as the rejected. The teacher did not answer this
    correction line, only said W: the pair teaches "after being corrected to
    W, say W", not the exact wording of the reply."""
    rec = turn["rec"]
    hist = rec.get("history") or []
    user0 = rec.get("user") or ""
    prompt = turn["prompt"]
    if (not hist or rec.get("quiet") or not user0.strip()
            or len(prompt) < 3 or prompt[-2]["role"] != "assistant"
            or prompt[-1]["role"] != "user"
            or user0 not in prompt[-1]["content"]):
        return None
    sp = dx.spoken_of(raw)
    prev_sp = dx.spoken_of(hist[-1].get("reply", ""))
    shared = sorted(_words_of(sp) & _words_of(prev_sp))
    if not shared:
        return None
    h = zlib.crc32((turn["id"] + raw).encode("utf-8"))
    w = shared[h % len(shared)]
    x = misspell(w, h >> 3)
    if x is None or x in sp or x in prompt[-1]["content"] \
            or w not in prompt[-2]["content"]:
        return None
    lines = CORRECTION_LINES[lang_of(user0)]
    line = lines[(h >> 5) % len(lines)].format(x=x, y=w)
    if ck.corrections(line) != [(x, w)] and \
            ck.corrections(line, x) != [(x, w)]:
        return None
    t2 = copy.deepcopy(turn)
    t2["prompt"][-1]["content"] = prompt[-1]["content"].replace(user0, line)
    t2["prompt"][-2]["content"] = prompt[-2]["content"].replace(w, x)
    t2["rec"]["user"] = line
    t2["rec"]["history"][-1]["reply"] = hist[-1].get("reply", "").replace(
        w, x)
    new = _swap_word(raw, w, x)
    if new == raw:
        return None
    info = {"x": x, "y": w, "source": "injected", "user_was": user0}
    return t2, raw, new, info


def correction_pair(turn, raw):
    """correction's two sources: a real correction line in the data, else
    the injected one (inject_correction)."""
    hist = turn["rec"].get("history") or []
    prev = dx.spoken_of(hist[-1].get("reply", "")) if hist else None
    got = correction(raw, turn["rec"]["user"], prev)
    if got:
        return turn, got[0], got[1], {"source": "real"}
    return inject_correction(turn, raw)


# --------------------------------------------------------------- pose copy

def pose_copy(raw, hist):
    """raw with its pose markers replaced, in place, by those of the most
    recent earlier reply in `hist` (raw replies, oldest first) that has at
    least two: the cross-turn collapse. None unless raw has two poses of
    its own, does not copy one of the last three itself, and the copy does
    (checker pose_copy). The speech and every other marker are untouched."""
    cur = [sp for sp in _spans(raw) if sp[2]["tag"] == pz.POSE]
    if len(cur) < 2 or not hist:
        return None
    prevs = [_rec(h) for h in hist[-ck.COPY_PREV:]]
    mine = ck.pose_seq(_rec(raw))
    if any(ck.copies(mine, ck.pose_seq(p)) for p in prevs):
        return None
    for src in reversed(hist[-ck.COPY_PREV:]):
        theirs = [sp for sp in _spans(src) if sp[2]["tag"] == pz.POSE]
        if len(theirs) < 2:
            continue
        out = raw
        for sp, tp in reversed(list(zip(cur, theirs))):
            part = _json_part(src, tp[0], tp[1])
            if part is None:
                return None
            try:
                data = json.loads(src[part[0]:part[1]])
            except ValueError:
                return None
            new = _marker_edit(out, sp, lambda d, data=data: dict(data))
            if new is False or new is None:
                return None
            out = new
        if out != raw and ck.copies(ck.pose_seq(_rec(out)),
                                    ck.pose_seq(_rec(src))):
            return out
    return None


def lang_of(line) -> str:
    """ja (kana in the line), zh (han without kana), else en."""
    s = line or ""
    if re.search(r"[\u3040-\u30ff]", s):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", s):
        return "zh"
    return "en"


def student_opening(paths, bodies=None) -> dict:
    """The most common exact opening ({variable: rounded value}) of the
    replies in these student.jsonl files (flow.checks' opening stats), or
    {} when none opens with a pose."""
    recs = []
    for p in paths:
        for _label, rs in ck.load(p):
            recs += rs
    _v, extra, _why = ck.check_opening(recs, bodies)
    return dict((extra.get("top_exact") or {}).get("opening") or {})


def opening_repeat(raw, user, top, offered=None):
    """raw with its first pose marker, a beat, as the student's common
    opening `top` ({variable: value}; the marker's dur / hold stay); None
    when the reply opens so already, has no such beat, the body lacks a
    variable of `top` (`offered`), or the swap changes the look the user
    asked for."""
    if not top or (offered is not None and not set(top) <= set(offered)):
        return None
    if ck.asked_at_me(user) and ("head.yaw" in top or "gaze.x" in top):
        return None
    old = ck.opening(_rec(raw))
    for sp in _spans(raw):
        t = sp[2]
        if t["tag"] != pz.POSE:
            continue
        tg = t["arg"]["targets"]
        if not dx.is_beat(tg.keys()):
            return None

        def swap(d):
            keep = {k: v for k, v in d.items() if k in ("dur", "hold")}
            return {**top, **keep}
        new = _marker_edit(raw, sp, swap)
        if new is False or new is None:
            return None
        got = ck.opening(_rec(new))
        if got is None or got == old or dict(got[1]) != {
                k: ck._round(v) for k, v in top.items()}:
            return None
        if ck.asked_side(user) and ck.first_turn(_rec(new)) \
                != ck.first_turn(_rec(raw)):
            return None
        return new
    return None


def _glances(raw, place) -> list:
    """The pose-marker spans among the first two that look toward `place`."""
    poses = [sp for sp in _spans(raw) if sp[2]["tag"] == pz.POSE][:2]
    return [sp for sp in poses
            if ck.looks_toward(sp[2]["arg"]["targets"], place)]


def glance_at_mention(raw, user, offered=None):
    """(chosen, rejected) for a line that puts something near her without
    asking her to look, or None. The teacher's reply has no glance in its
    first two beats: chosen is that reply with a glance beat (the pose
    marker of flow.checks.glance_targets: head and gaze toward the place,
    dur 0.3, hold 0.45) put first, rejected the reply as it is. It has one:
    chosen is the reply, rejected the reply with the signs of the glancing
    markers' head / gaze values reversed (not for "behind you", which has
    no opposite). Speech identical. `offered`: the variables the body has
    (the glance uses only those)."""
    place = ck.mentioned_place(user)
    if place is None:
        return None
    if not any(sp[2]["tag"] == pz.POSE for sp in _spans(raw)):
        return None
    want = ck.glance_targets(place)
    if offered is not None:
        want = {k: v for k, v in want.items() if k in offered}
    if not want:
        return None
    hit = _glances(raw, place)
    if not hit:
        marker = "[pose " + json.dumps(
            {**want, "dur": ck.GLANCE_DUR, "hold": ck.GLANCE_HOLD},
            ensure_ascii=False) + "]"
        return marker + raw, raw
    if place["behind"]:
        return None

    def reverse(d):
        for k in ("head.yaw", "gaze.x", "head.pitch", "gaze.y"):
            v = ck._num(d.get(k))
            if v is not None:
                d[k] = -v
        return d
    new = raw
    for sp in sorted(hit, key=lambda x: -x[0]):
        new = _marker_edit(new, sp, reverse)
        if new is False or new is None:
            return None
    if _glances(new, place):
        return None
    return raw, new


def synthetic_pair(kind, raw, user, has_hands=True, offered=None,
                   opening=None, turn=None):
    """The broken reply, a (chosen, rejected) pair (look_at_me), or a
    (turn', chosen, rejected, info) (correction: the turn's prompt may be
    edited); None when the kind does not apply."""
    if kind == "correction":
        return correction_pair(turn, raw) if turn is not None else None
    if kind == "pose_copy":
        hist = [h.get("reply", "") for h in
                (turn["rec"].get("history") or [])] if turn else []
        return pose_copy(raw, hist)
    if kind == "wave_frame":
        return wave_frame(raw)
    if kind == "opening_repeat":
        return opening_repeat(raw, user, opening, offered)
    if kind == "look_flip":
        return flip_look(raw, user)
    if kind == "look_at_me":
        return look_at_me(raw, user, offered)
    if kind == "glance_at_mention":
        return glance_at_mention(raw, user, offered)
    if kind == "hand_drop":
        return drop_hand(raw, user, has_hands)
    if kind == "template_open":
        return template_open(raw, user, offered)
    if kind == "beat_dup":
        return beat_dup(raw)
    raise ValueError(kind)


def synthetic_pairs(turns, bodies, per_kind, seed=0, offered=None,
                    kinds=KINDS, opening=None, kind_n=None) -> list:
    """[(kind, turn, intact, broken)]: up to per_kind per kind (kind_n:
    {kind: count} overrides it), over the turns in a seeded shuffle.
    `offered`: {avatar: variable names}; `opening`: the student's opening
    for opening_repeat (without one that kind makes none)."""
    rng = random.Random(seed)
    order = list(turns)
    rng.shuffle(order)
    out, n = [], Counter()
    cap = {k: (kind_n or {}).get(k, per_kind) for k in kinds}
    for t in order:
        for kind in kinds:
            if n[kind] >= cap[kind]:
                continue
            name = t["rec"]["avatar"]["name"]
            broken = synthetic_pair(kind, t["teacher"], t["rec"]["user"],
                                    bodies.has_hands(name),
                                    (offered or {}).get(name), opening, t)
            intact, tt = t["teacher"], t
            if isinstance(broken, tuple) and len(broken) == 2:
                intact, broken = broken          # look_at_me: chosen edited
            elif isinstance(broken, tuple):      # correction: prompt edited
                tt, intact, broken, info = broken
                tt = {**tt, "info": info}
            if broken is not None and broken != intact:
                out.append((kind, tt, intact, broken))
                n[kind] += 1
    return out


def offered_of(avatars, names) -> dict:
    """{avatar: variables its body offers} for the template check, from the
    body files (a VRM's too), whether or not anything is rendered."""
    out = {}
    for n in names:
        bm = avatars.body(n)
        if bm is not None:
            out[n] = pz.available(bm)
    return out


# ------------------------------------------------------------ calibration

def calibrate(pairs, ctx) -> dict:
    """Judge both sides of synthetic pairs and count, per kind, whether the
    judge prefers the intact reply: {kind: {n, na, vision: {win, tie, loss},
    dim: {name, win, tie, loss}}, ...} (module docstring)."""
    res = {}
    for kind, turn, intact, broken in pairs:
        both = judge_replies(turn, [("intact", intact), ("broken", broken)],
                             ctx)
        r = res.setdefault(kind, {"n": 0, "na": 0, "vision": Counter(),
                                  "dim": Counter(),
                                  "dim_name": KIND_DIM[kind]})
        a, b = both[0].get("scores"), both[1].get("scores")
        if a is None or b is None:
            r["na"] += 1
            continue
        r["n"] += 1
        look = kind == "look_flip"      # the text checker decides these
        d = vision_score(a, look) - vision_score(b, look)
        r["vision"]["win" if d > 0 else "loss" if d < 0 else "tie"] += 1
        dim = KIND_DIM[kind]
        x, y = a.get(dim), b.get(dim)
        if x is not None and y is not None:
            r["dim"]["win" if x > y else "loss" if x < y else "tie"] += 1
        else:
            r["dim"]["na"] += 1
        if ctx.stopped:
            break
    return {k: {**v, "vision": dict(v["vision"]), "dim": dict(v["dim"])}
            for k, v in res.items()}


def calibration_text(cal) -> str:
    """The calibration table: per kind, the pairs judged and how often the
    judge prefers the intact reply, overall (vision score) and on the
    kind's own dimension; rate = wins + half the ties over n."""
    def cell(c, n):
        w, t, l = c.get("win", 0), c.get("tie", 0), c.get("loss", 0)
        m = w + t + l
        return (f"{w:>3}/{t:>3}/{l:>3}  {(w + t / 2) / m:>5.0%}" if m
                else f"{'-':>16}")
    out = [f"{'kind':14s}{'n':>4}{'na':>4}  {'overall w/t/l  rate':<20}"
           f"{'own dimension w/t/l  rate'}"]
    for kind in KINDS:
        if kind not in cal:
            continue
        c = cal[kind]
        out.append(f"{kind:14s}{c['n']:>4}{c['na']:>4}  "
                   f"{cell(c['vision'], c['n']):<20}"
                   f"{cell(c['dim'], c['n'])}  ({c['dim_name']})")
    out.append("win: the judge scores the intact reply higher; tie: equal; "
               "loss: it prefers the broken one")
    return "\n".join(out)


# ------------------------------------------------------------------ output

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")


def build(turns, candidates, ctx, a, log=print):
    """Pairs from candidates {turn id: [(source, reply)]}: (rows, metas,
    drops Counter, stats). The teacher's reply joins the pool with
    a.teacher_chosen, as a chosen candidate only (never the rejected one)."""
    rows, metas, drops = [], [], Counter()
    nc = Counter()
    for t in turns:
        reps = list(candidates.get(t["id"], ()))
        if a.teacher_chosen:
            reps.append(("teacher", t["teacher"]))
        if len(reps) < 2:
            drops["no_candidates"] += 1
            continue
        if ctx.stopped:
            drops["judge_budget"] += 1
            continue
        cands = evaluate_turn(t, reps, ctx)
        nc["candidates"] += len(cands)
        nc["contradictions"] += sum(c["conflict"] == "contradiction"
                                    for c in cands)
        nc["visibility"] += sum(c["conflict"] == "visibility" for c in cands)
        ch, rj, why = choose(cands, a.min_gap, a.min_natural)
        if ch is None:
            drops[why] += 1
            continue
        rows.append(pair_row(t, ch["reply"], rj["reply"]))
        metas.append({"id": t["id"], "avatar": t["rec"]["avatar"]["name"],
                      "kind": "judge",
                      "chosen": cand_meta(ch), "rejected": cand_meta(rj),
                      "gap": round(ch["score"] - rj["score"], 3),
                      "candidates": len(cands)})
    return rows, metas, drops, nc


def synthetic_rows(pairs, bodies) -> tuple:
    rows, metas = [], []
    for kind, t, intact, broken in pairs:
        rows.append(pair_row(t, intact, broken))
        v = check_replies(t, [("intact", intact), ("broken", broken)], bodies)
        metas.append({"id": t["id"], "avatar": t["rec"]["avatar"]["name"],
                      "kind": f"synthetic:{kind}",
                      "lang": lang_of(t["rec"]["user"]),
                      **({"info": t["info"]} if t.get("info") else {}),
                      "chosen": {"src": "teacher" if intact == t["teacher"]
                                 else "teacher+edit",
                                 "verdicts": v[0]["verdicts"]},
                      "rejected": {"src": f"edit:{kind}",
                                   "verdicts": v[1]["verdicts"]}})
    return rows, metas


def summary_text(metas, cal, ctx, drops, nc, n_turns) -> str:
    out = [f"dpo: {len(metas)} pairs from {n_turns} turns"]
    kinds = Counter(m["kind"] for m in metas)
    avs = Counter(m["avatar"] for m in metas)
    out.append("by kind: " + (", ".join(f"{k} {v}" for k, v in sorted(
        kinds.items())) or "-"))
    langs = Counter((m["kind"], m["lang"]) for m in metas if "lang" in m)
    if langs:
        out.append("synthetic by user-line language: " + ", ".join(
            f"{k.split(':')[-1]}/{lg} {v}"
            for (k, lg), v in sorted(langs.items())))
    out.append("by avatar: " + (", ".join(f"{k} {v}" for k, v in sorted(
        avs.items())) or "-"))
    if drops:
        out.append("dropped turns: " + ", ".join(
            f"{k} {v}" for k, v in sorted(drops.items())))
    if nc:
        out.append(f"candidates {nc['candidates']}: {nc['contradictions']} "
                   f"left out as contradictions (text vs judge), "
                   f"{nc['visibility']} flagged invisible (never chosen)")
    if ctx.na:
        out.append("judge not applicable: " + ", ".join(
            f"{k} x{v}" for k, v in ctx.na.items()))
    if cal:
        out.append("judge calibration (synthetic pairs):\n"
                   + calibration_text(cal))
    if ctx.judge is not None:
        out.append(ctx.judge.report())
    if ctx.stopped:
        out.append("STOPPED: the judge call budget (--max-calls) ran out; "
                   "rerun with the same --judge-cache to continue for free")
    return "\n".join(out)


# --------------------------------------------------------------------- cli

def parse(argv):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Candidates: --k N samples through --backend, or "
               "--candidates FILE.jsonl.")
    ap.add_argument("--data", required=True,
                    help="a flow.dataset folder (r3)")
    ap.add_argument("--split", default=None,
                    help="a training run's split.json: only its --side "
                         "sessions (the held-out ones by default)")
    ap.add_argument("--side", default="eval", choices=("eval", "train"))
    ap.add_argument("--out", required=True, help="folder: dpo.jsonl, "
                    "dpo.meta.jsonl, summary.json, candidates.jsonl")
    ap.add_argument("--avatars", default=None, help="comma list; others are "
                    "left out")
    ap.add_argument("--limit", type=int, default=None, help="first N turns")
    ap.add_argument("--student-prompt", default="teacher",
                    choices=dx.STUDENT_PROMPTS,
                    help="the system prompt of the pairs, as dataexport sft "
                         "--student-prompt (short for a student trained "
                         "with it)")
    # What dataexport sft takes, so the turns and replies are the export's.
    ap.add_argument("--keep-lite", action="store_true",
                    help="also the turns of avatars generated without a rig "
                         "(live2d-lite), as dataexport sft --keep-lite")
    ap.add_argument("--no-correct-misses", dest="correct", default=True,
                    action="store_false")
    ap.add_argument("--no-manners", dest="manners_on", default=True,
                    action="store_false")
    ap.add_argument("--max-poses", type=int, default=dx.MAX_POSES)
    ap.add_argument("--max-beats", type=int, default=None)
    ap.add_argument("--max-hands", type=int, default=None)
    ap.add_argument("--lang-filter", default=None,
                    action=argparse.BooleanOptionalAction)
    ap.add_argument("--request-filter", default=None,
                    action=argparse.BooleanOptionalAction)
    src = ap.add_argument_group("candidates")
    src.add_argument("--k", type=int, default=0, help="student samples a "
                     "turn (needs --backend / --model; ollama, vllm ...)")
    src.add_argument("--candidates", action="append", default=[],
                     help="jsonl of {id, reply} (repeatable): replies "
                          "sampled elsewhere (an evalsft student.jsonl too)")
    src.add_argument("--conc", type=int, default=1,
                     help="requests / judge calls at once [1]")
    src.add_argument("--teacher-chosen", action="store_true",
                     help="the teacher's reply competes in each turn's pool")
    dr.add_args(ap)
    ap.set_defaults(backend="vllm")
    sel = ap.add_argument_group("choosing")
    sel.add_argument("--min-gap", type=float, default=MIN_GAP,
                     help=f"chosen - rejected score, at least [{MIN_GAP:g}]")
    sel.add_argument("--min-natural", type=int, default=0,
                     help="the chosen reply's judged natural score at "
                          "least this [0: off]")
    sel.add_argument("--no-judge", action="store_true",
                     help="text checks only")
    syn = ap.add_argument_group("minimal-edit pairs")
    syn.add_argument("--synthetic", type=int, default=0, metavar="N",
                     help="N minimal-edit pairs per kind "
                          f"({', '.join(KINDS)}) into the output")
    syn.add_argument("--kinds", default=None, help="comma list of the "
                     f"minimal-edit kinds to make [all: {', '.join(KINDS)}]")
    syn.add_argument("--kind-n", default=None, metavar="K=N,...",
                     help="a count per kind instead of --synthetic's, e.g. "
                          "look_flip=100,wave_frame=40")
    syn.add_argument("--opening-from", action="append", default=[],
                     metavar="STUDENT.JSONL",
                     help="an evalsft student.jsonl (repeatable): its most "
                          "common opening is what opening_repeat puts in "
                          "(flow.checks' opening stats)")
    syn.add_argument("--calibrate", type=int, default=0, metavar="N",
                     help="judge N minimal-edit pairs (balanced over the "
                          "kinds) and report agreement; with no candidates "
                          "that is all that runs")
    jg.add_judge_args(ap)
    jg.add_render_args(ap)
    ap.add_argument("--judge-dims", default=",".join(DIMS),
                    help="the judge fields that enter the pair score, comma "
                         f"list of {', '.join(jg.FIELDS)} [{','.join(DIMS)}]"
                         "; the others are recorded in the meta only")
    ap.add_argument("--sheets", default=None, help="folder to keep the "
                    "contact sheets of the judged replies in")
    a = ap.parse_args(argv)
    jg.SHOW_TARGETS = bool(getattr(a, "show_targets", False))
    global JUDGE_DIMS
    dims = tuple(x for x in a.judge_dims.split(",") if x)
    bad = [x for x in dims if x not in jg.FIELDS]
    if bad:
        ap.error(f"--judge-dims: unknown {', '.join(bad)} "
                 f"(of {', '.join(jg.FIELDS)})")
    JUDGE_DIMS = dims
    if a.k and not a.model and a.backend != "fake":
        ap.error("--k needs --model (the served student)")
    a.kind_list = tuple(x for x in (a.kinds or "").split(",") if x) or KINDS
    bad = [x for x in a.kind_list if x not in KINDS]
    if bad:
        ap.error(f"--kinds: unknown {', '.join(bad)} (of {', '.join(KINDS)})")
    a.kind_counts = {}
    for item in (a.kind_n or "").split(","):
        if not item:
            continue
        k, _, n = item.partition("=")
        if k not in KINDS or not n.isdigit():
            ap.error(f"--kind-n {item!r}: KIND=COUNT, KIND one of "
                     f"{', '.join(KINDS)}")
        a.kind_counts[k] = int(n)
    if a.kind_counts and not a.kinds:
        a.kind_list = tuple(k for k in KINDS if k in a.kind_counts)
    return a


def main(argv=None) -> int:
    a = parse(list(sys.argv[1:] if argv is None else argv))
    log = print
    use_judge = not a.no_judge and (a.k or a.candidates or a.calibrate)
    if a.renderer == "live2d" and use_judge:
        try:
            from ..l2dprobe.backend import ensure_display
        except ImportError:
            from l2dprobe.backend import ensure_display
        ensure_display()
    sessions = split_sessions(a.split, a.side) if a.split else None
    avs = [x for x in (a.avatars or "").split(",") if x] or None
    turns = load_turns(
        a.data, sessions, avs, a.limit, a.student_prompt, log=log,
        max_miss=dx.MAX_MISS, dup=dx.DUP_JACCARD, keep_lite=a.keep_lite,
        correct=a.correct,
        manners_on=a.manners_on, max_poses=a.max_poses,
        max_beats=a.max_beats, lang_filter=a.lang_filter,
        request_filter=a.request_filter, max_hands=a.max_hands)
    if not turns:
        raise SystemExit("no turns (the export keeps none of these "
                         "sessions)")
    bodies = ck.Bodies(a.data)
    bodies_av = jg.Avatars(a.data, "fake")
    avatars = jg.Avatars(a.data, a.renderer, a.models, log=lambda s: None,
                         squint=not a.no_smile_squint) \
        if use_judge else None
    judge = jg.judge_from_args(a, log=lambda s: None) if use_judge else None
    ctx = Context(bodies, avatars, judge, a.tiles, a.tile, a.follow, a.conc,
                  a.sheets, log)
    os.makedirs(a.out, exist_ok=True)
    rows, metas, cal = [], [], None
    drops, nc = Counter(), Counter()
    offered = offered_of(bodies_av, {t["rec"]["avatar"]["name"]
                                     for t in turns})
    try:
        # The judge is calibrated on the kinds it can see.
        cal_kinds = tuple(k for k in a.kind_list if k in CAL_KINDS)
        if a.calibrate and cal_kinds:
            per = max(1, -(-a.calibrate // len(cal_kinds)))
            pairs = synthetic_pairs(turns, bodies, per, a.seed, offered,
                                    cal_kinds)
            bykind = defaultdict(list)
            for p in pairs:
                bykind[p[0]].append(p)
            picked = []
            while len(picked) < a.calibrate and any(bykind.values()):
                for k in cal_kinds:
                    if bykind[k] and len(picked) < a.calibrate:
                        picked.append(bykind[k].pop(0))
            log(f"dpo: calibrating on {len(picked)} minimal-edit pairs "
                f"({Counter(p[0] for p in picked)})")
            cal = calibrate(picked, ctx)
            with open(os.path.join(a.out, "calibration.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(cal, fh, indent=1)
        cands = {}
        if a.candidates:
            cands = read_candidates(a.candidates)
        if a.k:
            fresh = sample(turns, a.k, a, a.conc, a.seed, log)
            for i, v in fresh.items():
                cands.setdefault(i, []).extend(v)
            write_jsonl(os.path.join(a.out, "candidates.jsonl"),
                        [{"id": i, "source": s, "reply": r}
                         for i, v in sorted(fresh.items()) for s, r in v])
        if cands:
            r_, m_, drops, nc = build(turns, cands, ctx, a, log)
            rows += r_
            metas += m_
        if a.synthetic or a.kind_counts:
            top = student_opening(a.opening_from, bodies) \
                if a.opening_from else None
            if "opening_repeat" in a.kind_list and not top:
                log("dpo: opening_repeat needs --opening-from "
                    "student.jsonl with poses; none made")
            sp = synthetic_pairs(turns, bodies, a.synthetic, a.seed + 1,
                                 offered, a.kind_list, top, a.kind_counts)
            log("dpo: minimal-edit pairs: " + ", ".join(
                f"{k} {sum(p[0] == k for p in sp)}" for k in a.kind_list))
            r_, m_ = synthetic_rows(sp, bodies)
            rows += r_
            metas += m_
    finally:
        if avatars is not None:
            avatars.close()
    write_jsonl(os.path.join(a.out, "dpo.jsonl"), rows)
    write_jsonl(os.path.join(a.out, "dpo.meta.jsonl"), metas)
    text = summary_text(metas, cal, ctx, drops, nc, len(turns))
    print(text)
    with open(os.path.join(a.out, "summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
