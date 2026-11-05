"""
Eye smile that shows: the happy squint a Cubism smile parameter needs.

On most Cubism rigs (Hiyori, Mao, Natori, Haru, Ren of the samples)
ParamEyeLSmile / ParamEyeRSmile is a reshaping of the *closing* eye: at the
open eye's rest value, sweeping smile 0 -> 1 moves no vertex, and only as
the eye closes does the lid take the arc (open 0.5 + smile 1 is a slight
happy squint, open 0 + smile 1 the closed happy arcs). Our body drives
eye.L.smile alone, so the model's commonest face beat, and the micro layer's
smile, rendered as the rest face on such rigs.

This module is the render-side fix, not a body change: the solver's targets,
the dataset trajectories and the SFT export all keep eye.*.smile as written.
Where Live2D parameters are produced for drawing (flow.stream's frame,
runtime's render loop, the judge's renderer), a smile s in 0..1 also closes
that eye part of the way:

    open = min(open, rest - K * s * (rest - closed))

so s = 0.8 at K = 0.6 is about half closed, a clear squint, and s = 0.3 a
gentle one; a blink closes further and the min lets it. It is applied after
the blink, so the blink is what the eye shows when it is lower.

Detection (needs): per side, whether the smile parameter has (almost) no
visible effect at the open eye's rest value but does at a partly closed eye,
measured on the Cubism Core (l2dprobe.core: no GL) as the summed per-drawable
vertex shift of smile 0 -> 1. It is computed lazily from the model3.json
(cached per file), not stored in the body file, so existing body files work;
no Core, no model or an unreadable moc is "unknown -> no coupling".
"""

from __future__ import annotations

import json
import os

import numpy as np

try:
    from ..l2dprobe import core as _core
except ImportError:
    try:
        from l2dprobe import core as _core
    except ImportError:                       # pragma: no cover
        _core = None

K = 0.6                  # share of the closing range a full smile takes
PART = 0.3               # the "partly closed" eye of the test, in open range
HIDDEN = 0.2             # effect at rest / effect partly closed below this
SEEN = 5e-4              # partly closed effect, of the rest mesh's extent

SIDES = ("L", "R")
_CACHE: dict = {}


def effect(m, smile: str, open_: str, o: float) -> float:
    """Summed per-drawable max vertex shift of smile (min -> max) with the
    eye open at o (everything else at its default)."""
    i = m.param_ids.index(smile)
    off = m.offsets
    m.set({open_: o, smile: float(m.param_min[i])})
    a = m.all_vertices()
    m.set({open_: o, smile: float(m.param_max[i])})
    b = m.all_vertices()
    d = abs(b - a).max(1)
    return float(sum(d[off[k]:off[k + 1]].max()
                     for k in range(len(m.drawable_ids))
                     if off[k + 1] > off[k]))


def measure(m, smile: str, open_: str) -> dict | None:
    """The side's numbers on a core model, or None when the rig lacks either
    parameter or its ranges are empty: {"rest", "closed" (open parameter at
    the eye's rest and closed values), "smile" ((neutral, full) values),
    "effect_rest", "effect_part", "extent", "needed"}."""
    if smile not in m.param_ids or open_ not in m.param_ids:
        return None
    i, j = m.param_ids.index(smile), m.param_ids.index(open_)
    rest, closed = float(m.param_default[j]), float(m.param_min[j])
    s0, s1 = float(m.param_default[i]), float(m.param_max[i])
    if not (rest > closed and s1 > s0):
        return None
    m.set({})
    ext = float(np.ptp(m.all_vertices(), 0).max())
    e_rest = effect(m, smile, open_, rest)
    e_part = effect(m, smile, open_, closed + PART * (rest - closed))
    return {"rest": rest, "closed": closed, "smile": (s0, s1),
            "effect_rest": e_rest, "effect_part": e_part, "extent": ext,
            "needed": needs(e_rest, e_part, ext)}


def needs(e_rest: float, e_part: float, extent: float) -> bool:
    """Smile is hidden on the open eye and shows on a partly closed one."""
    return e_part > SEEN * max(extent, 1e-9) and e_rest <= HIDDEN * e_part


def detect(model_json: str, ids: dict | None = None) -> dict:
    """{"L": side numbers, "R": ...} for the sides of the rig that need the
    coupling; {} when none does or it cannot be told. ids: {"eye.L.smile":
    pid, "eye.L.open": pid, ...} (the body file's declared parameters),
    else the standard ParamEye<L|R>Smile / Open."""
    key = (os.path.abspath(model_json), json.dumps(ids, sort_keys=True))
    if key in _CACHE:
        return _CACHE[key]
    out = {}
    try:
        if _core is None or not _core.available():
            raise ImportError("no Cubism Core")
        m = _core.load(model_json)
        for s in SIDES:
            sm = (ids or {}).get(f"eye.{s}.smile", f"ParamEye{s}Smile")
            op = (ids or {}).get(f"eye.{s}.open", f"ParamEye{s}Open")
            r = measure(m, sm, op)
            if r is not None and r["needed"]:
                out[s] = {"smile_id": sm, "open_id": op, **r}
    except (ImportError, OSError, KeyError, ValueError, RuntimeError):
        out = {}
    _CACHE[key] = out
    return out


def declared_ids(body_path: str | None) -> dict:
    """The eye.*.smile / eye.*.open parameter ids a body file declares (only
    those with the plain +1 scale)."""
    if not body_path or not os.path.exists(body_path):
        return {}
    try:
        with open(body_path, encoding="utf-8") as fh:
            dec = json.load(fh).get("declared", {})
    except (OSError, ValueError):
        return {}
    return {n: p for n, (p, s) in dec.items()
            if n.startswith("eye.") and n.endswith((".smile", ".open"))
            and abs(float(s) - 1.0) < 1e-6}


class Squint:
    """The coupling for the sides that need it. apply(over, dflt) returns
    the parameters with the eye open values lowered by the smile; no sides
    or k = 0 changes nothing."""

    def __init__(self, sides: dict, k: float = K):
        self.sides, self.k = dict(sides), float(k)

    def __bool__(self):
        return bool(self.sides) and self.k > 0

    def target(self, s: dict, smile: float) -> float:
        """The open value a smile (parameter units) allows."""
        s0, s1 = s["smile"]
        u = min(max((smile - s0) / (s1 - s0), 0.0), 1.0)
        return s["rest"] - self.k * u * (s["rest"] - s["closed"])

    def apply(self, over: dict, dflt: dict | None = None) -> dict:
        if not self:
            return over
        dflt = dflt or {}
        out = None
        for s in self.sides.values():
            sm = over.get(s["smile_id"])
            if sm is None:
                continue
            t = self.target(s, float(sm))
            cur = float(over.get(s["open_id"], dflt.get(s["open_id"],
                                                        s["rest"])))
            if t < cur:
                if out is None:
                    out = dict(over)
                out[s["open_id"]] = t
        return over if out is None else out


def for_model(model_json: str, body_path: str | None = None, k: float = K,
              log=None) -> Squint | None:
    """A Squint for the model, or None when no side needs it."""
    sides = detect(model_json, declared_ids(body_path))
    if not sides:
        return None
    if log:
        log("smile squint: eye." + "/".join(sorted(sides))
            + f" smile closes the eye (k {k:g}); render-side only")
    return Squint(sides, k)


def from_args(a, model_json: str, body_path: str | None = None, log=None):
    """For parsed arguments: None under --no-smile-squint."""
    if getattr(a, "no_smile_squint", False):
        return None
    return for_model(model_json, body_path, K, log)


def add_args(ap) -> None:
    """--no-smile-squint, for flow.director's, flow.runtime's and
    flow.judge's parsers (once, when they share one)."""
    if "--no-smile-squint" in ap._option_string_actions:
        return
    ap.add_argument("--no-smile-squint", action="store_true",
                    help="do not close the eye with its smile (flow.squint): "
                         "on rigs whose smile only shows on a closing eye "
                         "the smile is then invisible")
