#!/usr/bin/env python3
"""
Repair layer for the live loop (flow.live --repair auto|log|off): the
student's markers are run through the checkers of flow.checks while the
reply streams, and a checkable failure gets a deterministic fix before the
sentence is planned. No model is called (docs/research.md section 4: validate,
then fix; re-asking costs a second generation, which live speech cannot
afford).

A reply streams sentence by sentence (flow.live.SentenceCutter). A Turn
holds what was already performed (the repaired sentences, kept as spoken)
and sees each new sentence with that prefix: the checker is run on
prefix + sentence, as it would be on the reply so far, and the fix may touch
the current sentence only. A failure whose marker lies in a sentence that is
already spoken is logged ("already spoken"), never edited.

  check              fix                                               fixes?
  look_sign          the first turning marker of the sentence: head.yaw   yes
                     and gaze.x get the asked side's sign (her right is
                     negative), magnitudes kept. A reply that never turns
                     is only logged at the end.
  look_at_me         every head.yaw >= 7 degrees / gaze.x >= 0.3 (the        yes
                     checker's thresholds) in the sentence's pose markers
                     becomes 0, while the reply so far ends sideways.
  frame_for_dance    [frame full] right before a whole-body move, when the    yes
                     reply starts in bust, the body switches camera and no
                     full frame came before. A later [frame full] of the
                     same sentence is removed (once per reply).
  frame_before_wave  A [frame full] before a wave is removed (the prompt:     yes
                     "never for a wave", it stays in bust), when no
                     whole-body move sits between the frame and the wave
                     (then the frame is the dance's; only logged). The
                     checker FAILS on a full frame before a wave, so this
                     is the opposite of frame_for_dance.
  wave_obeyed        An explicit wave request, a body whose hands move and   yes
                     [move wave] on the menu, and no wave or hand pose in
                     the FIRST sentence: [move {"name": "wave"}] is put at
                     its start (decided at sentence 1: the later ones are
                     not known yet; a wave in a later sentence too is a
                     second wave). Otherwise logged at the end.
  body_request_obeyed  Logged at the end only (which hand action, and        log
                     where, is not unambiguous).
  glance_at_mention  Not here: flow.live's glance reflex does it.            -

--repair log checks and logs everything and changes nothing; off does
neither. Every repair is a line "repair: turn N.k <check>: <before> ->
<after>", a {"kind": "repair", ...} record of turns.jsonl, and, when the raw
whole reply fails the check and the repaired one passes it, a DPO pair
(pair(), flow.dpo's schema: the prompt as sent, chosen = the repaired reply,
rejected = the model's raw reply, meta kind repair_<check>, src live) in
<record>/repair_pairs.jsonl with repair_pairs.meta.jsonl beside it, the
layout scripts/train_dpo.py reads with --data.

The stored history is the reply as she performed it, the repaired one
(rebuild()): she must not be shown, on the next turn, a look she never
made, or a fix she then copies the old way. The raw reply is kept in the
turn's record (reply_unrepaired) and is the pair's rejected side.
"""

from __future__ import annotations

import json
import os

MODES = ("auto", "log", "off")
DEFAULT_MODE = "auto"

# The checks whose repaired reply makes a DPO pair.
PAIRED = ("look_sign", "look_at_me", "frame_for_dance",
          "frame_before_wave", "wave_obeyed")
WAVE_MARKER = '[move {"name": "wave"}]'


def _ck():
    """flow.checks, imported when first needed (its chain is heavy)."""
    try:
        from . import checks as ck
    except ImportError:
        from flow import checks as ck
    return ck


def _dx():
    try:
        from . import dataexport as dx
    except ImportError:
        from flow import dataexport as dx
    return dx


def _fr():
    try:
        from . import framing as fr
    except ImportError:
        from flow import framing as fr
    return fr


def _fullwidth(text):
    try:
        from . import director as dr
    except ImportError:
        from flow import director as dr
    return text.translate(dr._FULLWIDTH)


# ------------------------------------------------------------ the edits

def _json_part(text, a, b):
    i, j = text.find("{", a, b), text.rfind("}", a, b)
    return (i, j + 1) if 0 <= i < j else None


def _cut(text, a, b):
    """text without text[a:b], a marker, and the space after it when a space
    (or the start) is before it, so no double space is left."""
    if b < len(text) and text[b] == " " and (a == 0 or text[a - 1] in " \n"):
        b += 1
    return text[:a] + text[b:]


def edit_marker(text, span, fn):
    """text with the pose marker at span (dataexport.marker_spans') rewritten:
    fn gets the marker's JSON as an ordered dict and edits it. None when the
    JSON is not strict (a lenient marker is not edited)."""
    a, b, _t = span
    part = _json_part(text, a, b)
    if part is None:
        return None
    try:
        data = json.loads(text[part[0]:part[1]])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    new = fn(dict(data)) or data
    return text[:part[0]] + json.dumps(new, ensure_ascii=False) \
        + text[part[1]:]


def _num(x):
    return float(x) if isinstance(x, (int, float)) \
        and not isinstance(x, bool) else None


def _is_pose(t):
    return t["tag"] == "pose"


def _is_full(t):
    return t["tag"] == "frame" and t["arg"] == "full"


def _is_move(t, names):
    return t["tag"] == "move" and (t["arg"] or {}).get("name") in names


def _marker_at(text, start):
    """The marker text that opens at text[start] (to its closing bracket)."""
    end = text.find("]", start)
    return text[start:end + 1] if end >= 0 else text[start:]


class _Body:
    """What the checkers ask of the body: the live body, not a file."""

    def __init__(self, hands=True, frames=True):
        self.hands, self.frames = hands, frames

    def has_hands(self, _avatar):
        return self.hands

    def can_frame(self, _avatar):
        return self.frames


def hands_of(bm) -> bool:
    """Whether the body's hands reach anything (flow.framing.budget:
    hand_moves full > 0), as flow.checks.Bodies.has_hands."""
    try:
        b = _fr().budget(bm)
    except Exception:  # noqa: BLE001 - an unmeasured body: assume hands
        return True
    return bool((b.get("hand_moves") or {}).get("full", 1))


# ----------------------------------------------------------------- Turn

class Turn:
    """The repairs of one reply, sentence by sentence (module docstring).

    user      the viewer line(s) of the turn
    mode      auto | log | off
    start     the framing the reply starts in ("bust" / "full"), None when
              the prompt has no [frame] markers
    frames    the body can switch the camera (flow.framing.can_frame)
    hands     the body's hands reach something
    wave      [move wave] is on the menu and the parser takes it
    quiet     a quiet-chat turn: nothing is repaired
    """

    def __init__(self, user, mode=DEFAULT_MODE, avatar="?", start=None,
                 frames=False, hands=True, wave=False, quiet=False,
                 turn=0):
        self.user, self.mode, self.avatar = user or "", mode, avatar
        self.start, self.turn, self.quiet = start, int(turn), bool(quiet)
        self.frames = bool(frames and start is not None)
        self.hands, self.wave = bool(hands), bool(wave)
        self.on = mode in ("auto", "log") and not quiet
        self.body = _Body(hands, bool(frames))
        self.done = []            # the sentences as performed (raw: log)
        self.pairs = []           # (original, performed) per sentence
        self.fixes = []           # every fix, in order
        self.k = 0
        self._told = set()        # (check, why) logged once per reply

    # -- the records the checkers read ---------------------------------

    def rec(self, text):
        return _ck().make_rec(text, self.user, self.avatar, "live",
                              self.turn or 1, start=self.start,
                              quiet=self.quiet)

    def verdict(self, check, text):
        v, _e, why = _ck().CHECKS[check]([self.rec(text)], self.body)
        rid = next(iter(v))
        return v[rid], why.get(rid)

    def prefix(self) -> str:
        return "".join(self.done)

    @property
    def changed(self) -> bool:
        return any(f["applied"] for f in self.fixes)

    # -- fixes ----------------------------------------------------------

    def _fix(self, check, before, after, ok=True, why=None):
        f = {"check": check, "k": self.k, "before": before, "after": after,
             "applied": bool(ok and self.mode == "auto"), "ok": bool(ok)}
        if why:
            f["why"] = why
        self.fixes.append(f)
        return f

    def _unfixed(self, check, before, why, once=True):
        if once and (check, why) in self._told:
            return None
        self._told.add((check, why))
        return self._fix(check, before, before, False, why)

    # -- one sentence ---------------------------------------------------

    def sentence(self, raw, k=None):
        """(the sentence to plan, [fixes of it]). The sentence is returned
        unchanged in log mode and when nothing applies."""
        self.k = self.k + 1 if k is None else int(k)
        first = not self.pairs
        mark = len(self.fixes)
        new = raw
        if self.on:
            new = self._look_sign(new)
            new = self._look_at_me(new)
            new = self._frame_wave(new)
            new = self._frame_dance(new)
            if first:
                new = self._wave(new)
        out = new if self.mode == "auto" else raw
        self.done.append(out)
        self.pairs.append((raw, out))
        return out, self.fixes[mark:]

    def skip(self, raw):
        """A sentence that was not planned (dropped trailing markers):
        the history keeps it as it was."""
        self.done.append(raw)
        self.pairs.append((raw, raw))

    def _look_sign(self, new):
        ck, dx = _ck(), _dx()
        side = ck.asked_side(self.user)
        if side is None:
            return new
        want = -1 if side == "R" else 1
        pre = self.prefix()
        if ck.first_turn(self.rec(pre)) is not None:
            return new                    # decided in a spoken sentence
        got = ck.first_turn(self.rec(pre + new))
        if got is None or got == want:
            return new
        sp = None
        for s in dx.marker_spans(new):
            tg = s[2]["arg"]["targets"] if _is_pose(s[2]) else {}
            yaw, gx = _num(tg.get("head.yaw")), _num(tg.get("gaze.x"))
            if (yaw is not None and abs(yaw) >= ck.YAW_MIN) or \
                    (gx is not None and abs(gx) >= ck.GAZE_MIN):
                sp = s
                break
        if sp is None:
            return new
        before = new[sp[0]:sp[1]]

        def side_to(d):
            for key in ("head.yaw", "gaze.x"):
                v = _num(d.get(key))
                if v is not None and v != 0:
                    d[key] = want * abs(d[key])
            return d
        out = edit_marker(new, sp, side_to)
        if out is None or ck.first_turn(self.rec(pre + out)) != want:
            self._unfixed("look_sign", before, "the marker is not strict "
                          "JSON")
            return new
        self._fix("look_sign", before, _marker_at(out, sp[0]))
        return out

    def _look_at_me(self, new):
        ck, dx = _ck(), _dx()
        if not ck.asked_at_me(self.user):
            return new
        pre = self.prefix()
        if self.verdict("look_at_me", pre + new)[0] != ck.FAIL:
            return new
        out, before, after = new, [], []
        # Back to front: the spans before the edit stay where they are.
        for sp in reversed([s for s in dx.marker_spans(new)
                            if _is_pose(s[2])]):
            tg = sp[2]["arg"]["targets"]
            bad = [key for key, lim in (("head.yaw", ck.YAW_MIN),
                                        ("gaze.x", ck.GAZE_MIN))
                   if _num(tg.get(key)) is not None
                   and abs(_num(tg[key])) >= lim]
            if not bad:
                continue

            def centre(d, bad=bad):
                for key in bad:
                    d[key] = 0
                return d
            e = edit_marker(out, sp, centre)
            if e is None:
                self._unfixed("look_at_me", new[sp[0]:sp[1]],
                              "the marker is not strict JSON")
                continue
            before.insert(0, new[sp[0]:sp[1]])
            after.insert(0, _marker_at(e, sp[0]))
            out = e
        if out == new:
            if not before:
                self._unfixed("look_at_me", "the reply so far ends "
                              "sideways", "already spoken")
            return new
        if self.verdict("look_at_me", pre + out)[0] != ck.PASS:
            self._unfixed("look_at_me", " ".join(before), "still sideways "
                          "after the fix")
            return new
        self._fix("look_at_me", " ".join(before), " ".join(after))
        return out

    def _frame_dance(self, new):
        ck, dx, fr = _ck(), _dx(), _fr()
        if not self.frames or self.start == "full":
            return new
        pre = self.prefix()
        if any(_is_move(s[2], fr.WHOLE_BODY) for s in dx.marker_spans(pre)):
            return new                    # decided in a spoken sentence
        spans = dx.marker_spans(new)
        i = next((i for i, s in enumerate(spans)
                  if _is_move(s[2], fr.WHOLE_BODY)), None)
        if i is None or self.verdict("frame_for_dance", pre + new)[0] \
                != ck.FAIL:
            return new
        a, b = spans[i][0], spans[i][1]
        before = new[a:b]
        out = new
        # A full frame later in this sentence moves up to the dance.
        for s in reversed(spans[i + 1:]):
            if _is_full(s[2]):
                out = _cut(out, s[0], s[1])
        pad = "" if a == 0 or out[a - 1].isspace() else " "
        out = out[:a] + pad + "[frame full] " + out[a:]
        if self.verdict("frame_for_dance", pre + out)[0] != ck.PASS:
            return new
        self._fix("frame_for_dance", before, "[frame full] " + before)
        return out

    def _frame_wave(self, new):
        ck, dx, fr = _ck(), _dx(), _fr()
        if not self.frames:
            return new
        pre = self.prefix()
        pre_spans = dx.marker_spans(pre)
        if any(_is_move(s[2], ("wave",)) for s in pre_spans):
            return new                    # decided in a spoken sentence
        spans = dx.marker_spans(new)
        i = next((i for i, s in enumerate(spans)
                  if _is_move(s[2], ("wave",))), None)
        if i is None or self.verdict("frame_before_wave", pre + new)[0] \
                != ck.FAIL:
            return new
        before = "[frame full] " + new[spans[i][0]:spans[i][1]]
        if any(_is_full(s[2]) for s in pre_spans):
            self._unfixed("frame_before_wave", before, "already spoken")
            return new
        full = [s for s in spans[:i] if _is_full(s[2])]
        whole = any(_is_move(s[2], fr.WHOLE_BODY)
                    for s in pre_spans + spans[:i])
        if whole or not full:
            self._unfixed("frame_before_wave", before, "the frame belongs "
                          "to a whole-body move")
            return new
        out = new
        for s in reversed(full):
            out = _cut(out, s[0], s[1])
        if self.verdict("frame_before_wave", pre + out)[0] != ck.PASS:
            return new
        self._fix("frame_before_wave", before, before[len("[frame full] "):])
        return out

    def _wave(self, new):
        ck = _ck()
        if not (ck.asked_wave(self.user) and self.hands and self.wave) \
                or ck.has_wave(self.rec(new)):
            return new
        lead = len(new) - len(new.lstrip())
        out = new[:lead] + WAVE_MARKER + " " + new[lead:]
        if self.verdict("wave_obeyed", out)[0] != ck.PASS:
            return new
        self._fix("wave_obeyed", "no wave", WAVE_MARKER)
        return out

    # -- the end of the reply -------------------------------------------

    def finish(self):
        """The failures nothing repaired, logged once at the end of the
        reply, from the whole reply as performed: [fixes]."""
        if not self.on:
            return []
        ck = _ck()
        mark = len(self.fixes)
        rec = self.rec(self.prefix())
        v, _e, why = ck.check_look_sign([rec], self.body)
        if v[rec["id"]] == ck.FAIL and not any(
                f["check"] == "look_sign" for f in self.fixes):
            reason = why.get(rec["id"])
            self._unfixed("look_sign", "the reply never turns the right way"
                          if reason == "no_turn" else "wrong side",
                          "no marker to repair" if reason == "no_turn"
                          else "already spoken")
        if ck.asked_wave(self.user) and self.hands and not ck.has_wave(rec):
            self._unfixed("wave_obeyed", "no wave in the reply",
                          "wave decided at the first sentence"
                          if self.wave else "no wave move on this body")
        if self.hands and ck.asked_hands(self.user) \
                and not ck.asked_wave(self.user) and not ck.has_hand(rec):
            self._unfixed("body_request_obeyed", "no hand action in the "
                          "reply", "which action is not unambiguous")
        return self.fixes[mark:]

    # -- the reply as performed -----------------------------------------

    def rebuild(self, reply) -> str:
        """The model's whole reply with each repaired sentence replaced, in
        order, each searched from where the last one ended (the cutter
        turned full-width brackets into ASCII ones: the reply is turned
        the same way first). The sentences joined when one is not found."""
        if not self.changed:
            return reply
        text = _fullwidth(reply)
        out, cur = [], 0
        for orig, new in self.pairs:
            if orig == new:
                continue
            i = text.find(orig, cur)
            if i < 0:
                return "".join(new for _o, new in self.pairs)
            out.append(text[cur:i])
            out.append(new)
            cur = i + len(orig)
        out.append(text[cur:])
        return "".join(out)

    def verdicts(self, raw_reply, repaired) -> dict:
        """{check: (raw verdict, repaired verdict)} on the whole replies for
        the checks a fix of this reply touched."""
        return {c: (self.verdict(c, raw_reply)[0],
                    self.verdict(c, repaired)[0])
                for c in dict.fromkeys(f["check"] for f in self.fixes
                                       if f["applied"]) if c in PAIRED}

    def pair_check(self, raw_reply, repaired):
        """The first applied check whose raw whole reply fails and whose
        repaired one passes (the pair's kind), with all verdicts; (None,
        verdicts) when none does: a repair of a prefix that the rest of
        the reply would have mended is no pair."""
        ck = _ck()
        v = self.verdicts(raw_reply, repaired)
        for c, (a, b) in v.items():
            if a == ck.FAIL and b == ck.PASS:
                return c, v
        return None, v


# -------------------------------------------------------------- the lines

def log_line(turn, f) -> str:
    """The log line of a fix: "repair: turn N.k <check>: <before> ->
    <after>"; one that was not applied says why."""
    head = f"repair: turn {turn}.{f['k']} {f['check']}: "
    if not f["ok"]:
        return head + f"{f['before']} (not repaired: {f.get('why')})"
    return head + f"{f['before']} -> {f['after']}" + (
        "" if f["applied"] else " (log mode: not applied)")


def record(turn, user, f, mode) -> dict:
    """The {"kind": "repair", ...} line of turns.jsonl."""
    return {"kind": "repair", "turn": turn, "k": f["k"], "check": f["check"],
            "before": f["before"], "after": f["after"],
            "applied": f["applied"], "repaired": f["ok"], "mode": mode,
            **({"why": f["why"]} if f.get("why") else {}), "user": user}


# ----------------------------------------------------------------- pairs

def pair(prompt, raw, repaired, check, meta=None):
    """(row, meta) of one DPO pair in flow.dpo's schema: the prompt as chat
    messages, chosen = the repaired reply, rejected = the model's raw one;
    meta kind repair_<check>, src live."""
    row = {"prompt": [{"role": m["role"], "content": m["content"]}
                      for m in prompt],
           "chosen": [{"role": "assistant", "content": repaired}],
           "rejected": [{"role": "assistant", "content": raw}]}
    return row, {**(meta or {}), "kind": f"repair_{check}", "src": "live"}


def write_pair(folder, row, meta):
    """Append a pair to <folder>/repair_pairs.jsonl and its meta to
    repair_pairs.meta.jsonl (scripts/train_dpo.py --data reads both, line
    for line)."""
    os.makedirs(folder, exist_ok=True)
    for name, obj in (("repair_pairs.jsonl", row),
                      ("repair_pairs.meta.jsonl", meta)):
        with open(os.path.join(folder, name), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
