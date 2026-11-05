#!/usr/bin/env python3
"""
A vision-language judge for a student's replies: the avatar as the viewer
sees it, scored by a VLM. No training code; the DPO pair builder
(scripts/dpo_pairs.py, flow.dpo) combines this with flow.checks.

1. Keyframes (walk): a reply's markers are parsed with the runtime's parser
   (pose.parser_for(bm, moves=True, frames=True)), timed as the offline
   director times them (director.timeline, no TTS), solved (pose.
   solve_poses, moves.solve_moves) and played frame by frame through the
   same layers the dataset and the runtime use (moves.MoveLayer, pose.
   PoseLayer with head carry, pose ease and the expression baseline, and an
   end-of-speech release), so the state of one marker carries into the next
   as it does live: a head angle stays until a marker sets it again, the
   face keeps its baseline, a hand comes back. One still per pose / move
   marker is taken at the marker's peak (the end of its move, the hold
   begins): for a pose at start + the longest eased move of its parameters,
   for a [move] the frame of the move's span where its parameters are
   furthest from rest (a wave's raised hand, a nod's lowest head). A
   [frame] marker sets the camera of the stills after it (framing.
   frame_tokens), the camera follows a whole-body move in bust as the
   runtime's does (framing.follow_tokens), and a [frame] marker nothing
   follows gets a still of its own. The turn starts in the record's framing
   (home) and in the record's body state (the "(your body now: ...)" line:
   head, face and body angles solved to the rest of the layers; hands and
   arm drawings are left at rest) when known. A first still of the start
   state is always there for reference. Not rendered: lip movement (the
   mouth stays at rest), the idle's drift, physics.
2. Renderers: Live2DRenderer (l2dprobe Live2DBackend, one per avatar, reused
   across replies; needs GL: xvfb-run -a), FakeRenderer (no GL, for tests
   and --renderer fake). VRM is not supported: the browser page needs a
   whole recorded state per frame (flow.vrmbody) and a headless Chromium;
   the judge returns NA for a VRM (and for an avatar whose Live2D model is
   not on this machine).
3. Contact sheet (compose): the stills left to right, at most --tiles (6),
   each --tile px (256) with a caption: its number and camera, the first
   words the marker precedes, its key targets. At most 6 x 256 px keeps the
   image around 600-900 tokens.
4. The judge (Judge): the sheet, the viewer's message with its tail lines,
   the spoken text and the body's motion budget in one request to a VLM
   (OpenAI-compatible: siliconflow, vllm; anthropic; fake for tests),
   temperature 0, JSON back (request_done 0-2, visible 0-2, expression_fit
   0-2, natural 1-5, issues), parsed leniently, one retry on bad JSON,
   cached on disk by (sheet hash, prompt hash, model), --max-calls.

    python -m flow.judge render --data data/datagen/r3 \\
        --id Hiyori.cheerful.0003/t02 --out sheet.png
    python -m flow.judge score --data data/datagen/r3 --id ... \\
        --judge-backend vllm --judge-model Qwen/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import threading

import numpy as np

try:
    from . import body as fb
    from . import dataexport as dx
    from . import dataset as ds
    from . import director as dr
    from . import framing as fr
    from . import handask as ha
    from . import moves as mv
    from . import pose as pz
    from . import squint as sq
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from flow import body as fb
    from flow import dataexport as dx
    from flow import dataset as ds
    from flow import director as dr
    from flow import framing as fr
    from flow import handask as ha
    from flow import moves as mv
    from flow import pose as pz
    from flow import squint as sq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30.0
TAIL = 0.5               # seconds stepped after the last marker's end
RENDER_SIZE = 512        # the backend draws at this size; tiles are scaled
# 320 px x 5 tiles (rest + four stills): about the image tokens of the
# 256 x 6 sheet of before, with a face that reads in a small head turn.
TILE = 320
MAX_TILES = 5
# The judge is shown the markers' targets (captions and the tile list) only
# with --show-targets. Blind by default: the targets say what the frames
# should show, so a judge that reads them can score a flipped look or a
# missing wave without looking; the text checkers already cover what the
# targets say.
SHOW_TARGETS = False
WORDS = 5                # words the marker precedes shown in a caption
CJK_CHARS = 9
BG = (206, 208, 214)
MODEL_ROOTS = ("live2d-models", "CubismWebSamples")


def _quiet(_s):
    pass


# ----------------------------------------------------------- the start state

_NOW = re.compile(r"^\(your body now: (.*)\)\s*$", re.M)
_HAND = re.compile(r"hand\.[LR] \([^)]*\)")
_ITEM = re.compile(r"^([a-z][\w.]*) (-?\d+(?:\.\d+)?)$")
_CAMERA = re.compile(r"^\(camera: (bust|full)\b", re.M)


def now_targets(user_content) -> dict:
    """{variable: value} of the "(your body now: ...)" line of a user turn:
    what is away from rest. The hands' positions and "arms in another
    drawing" are left out (the hands come from the layers; a still starts
    them at rest)."""
    m = _NOW.search(user_content or "")
    if not m:
        return {}
    out = {}
    for item in _HAND.sub("", m.group(1)).split(","):
        g = _ITEM.match(item.strip())
        if g:
            out[g.group(1)] = float(g.group(2))
    return out


def home_of(rec) -> str:
    """The framing a record's turn starts in: its framing.start, else its
    camera line, else full."""
    f = rec.get("framing")
    if isinstance(f, dict) and f.get("start") in fr.FRAMES:
        return f["start"]
    m = _CAMERA.search(rec.get("user_content") or "")
    return m.group(1) if m else "full"


# ------------------------------------------------------------ the keyframes

def carry_roles(bm) -> dict:
    """{role: (parameter, 1.0, None)}: the parameter of each head and gaze
    angle (pose.CARRY_ROLES) that moves it most, as runtime's roles give
    pose.tokens (carry needs them: a body-only run has no rig)."""
    want = {r: v for r, v in pz.CARRY_ROLES.items() if v in bm.fidx}
    movers = pz._main_movers(bm, 0, {v: 0.0 for v in want.values()},
                             bm.solvable(0))
    return {r: (movers[v], 1.0, None) for r, v in want.items()
            if v in movers}


def _short(v):
    if isinstance(v, str):
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.0f}"


def _targets_text(targets, limit=4) -> str:
    items = [f"{k} {_short(v)}" for k, v in targets.items()]
    more = len(items) - limit
    return ", ".join(items[:limit]) + (f" +{more}" if more > 0 else "")


def _words_after(spoken, at, n=WORDS) -> str:
    """The first words of the speech from `at`: n words, or CJK_CHARS
    characters of an unspaced script."""
    s = (spoken or "")[int(at or 0):].strip()
    if not s:
        return ""
    cjk = sum(1 for c in s[:12] if dr._is_cjk(c))
    if cjk >= 4:
        return s[:CJK_CHARS] + ("..." if len(s) > CJK_CHARS else "")
    w = s.split()
    return " ".join(w[:n]) + ("..." if len(w) > n else "")


class Walk(dict):
    """walk()'s result: keys (the stills, in marker order; keys[0] is the
    start state), spoken, home, tags, warnings, start (the body-now
    targets), end (speech seconds)."""


def walk(bm, raw, home="bust", now=None, seed=0, follow=True, fps=FPS,
         log=_quiet) -> Walk:
    """The stills of a reply on a body (module docstring, 1). `raw`: the
    reply with its markers; `home`: the framing the turn starts in; `now`:
    {variable: value} the body starts in (now_targets) or None for rest.
    Each key: {kind: rest | pose | move | frame, n, t (peak, seconds), at
    (offset in the speech), words, targets, name (a move), view, followed
    (the camera followed a move), params, parts, caption, ok}."""
    framed = fr.can_frame(bm)
    home = home if framed and home in fr.FRAMES else "full"
    parser = pz.parser_for(bm, moves=True, frames=framed)
    parser.feed(raw or "")
    parser.close()
    spoken, tags = parser.spoken, parser.tags
    acts, _spans, end = dr.timeline(spoken, tags, dr.CJK_RATE, dr.LATIN_RATE)
    rest = bm.default_values()
    arms = pz.arm_params(bm)
    base = dict(rest)
    start = {k: v for k, v in (now or {}).items() if k in bm.fidx}
    if start:
        res = bm.solve(start)
        base.update({p: float(x) for p, x in res["params"].items()
                     if p not in arms})
    # Every part opacity the rest drawing needs, not only the ones the
    # measurement set (a rig without pose3 draws both arms otherwise), as
    # body._render and solve_poses start.
    rest_parts = {**bm.blob.get("rest_parts", {}),
                  **bm.variants[bm.match_variant(base)]["parts"]}
    roles = carry_roles(bm)
    ang = pz.angle_params(list(bm.params), roles)
    poses = pz.solve_poses(bm, acts, current={
        p: x for p, x in base.items() if x != rest.get(p)}, log=log) \
        if any(a["tag"] == pz.POSE for a in acts) else []
    poses = [p for p in poses if not p.get("via")]
    mvs = mv.solve_moves(bm, acts, log=log, seed=seed) \
        if any(a["tag"] == mv.MOVE for a in acts) else []
    toks = pz.tokens(poses, roles, arms, carry=True, baseline=True)
    if poses:
        toks.append(pz.release_token(end))
    toks_m = mv.tokens(mvs, arms)
    # The camera: the model's [frame] markers, and the follow of
    # whole-body moves in bust.
    ftoks = fr.frame_tokens(acts)
    if follow and framed:
        ftoks += fr.follow_tokens(mvs, ftoks, home, log=log)
    ftoks.sort(key=lambda k: float(k["t"]))

    def view_at(t):
        cur, was = home, False
        for k in ftoks:
            if float(k["t"]) <= t:
                cur, was = k["view"], bool(k.get("follow"))
        return cur, was

    seconds = max(end, pz.end_time(poses, True), mv.end_time(mvs)) + TAIL
    n = int(math.ceil(seconds * fps)) + 1
    mlay = mv.layer_for(toks_m, fps, dict(base), dict(rest_parts))
    play = pz.layer_for(toks, fps, dict(base), dict(rest_parts), ease=True,
                        angles=ang, baseline=1.0)
    bounds = {p: (float(e["min"]), float(e["max"]))
              for p, e in bm.params.items()}
    snaps = []
    for f in range(n):
        over, parts = dict(base), dict(rest_parts)
        still = ()
        if mlay is not None and mlay.busy():
            over, parts = mlay.apply(f, over, parts)
            still = mlay.holding
        if play is not None:
            over, parts = play.apply(f, over, parts, still)
            if play.has_base():
                over = play.base_apply(f, over, bounds)
        snaps.append((over, dict(parts or {})))

    def snap(f):
        over, parts = snaps[min(max(int(f), 0), n - 1)]
        return dict(over), dict(parts)

    def span(p):
        e = bm.params[p]
        return max(float(e["max"]) - float(e["min"]), 1e-9)

    keys = [{"kind": "rest", "n": 0, "t": 0.0, "at": 0, "words": "",
             "targets": {}, "name": None, "view": home, "followed": False,
             "params": dict(base), "parts": dict(rest_parts),
             "caption": ["rest" + (" (start)" if start else "")], "ok": True}]
    body_keys = []
    for p in poses:
        act = acts[p["act"]]
        angle = lambda q: q in ang  # noqa: E731
        dmax = max([pz.eased(p["dur"], p["dur"] + p["hold"], p["release"],
                             angle(q), fps, p["start"])[0]
                    for q in p["params"]] or [p["dur"]])
        t = p["start"] + dmax
        body_keys.append((p["act"], {
            "kind": "pose", "t": round(t, 3), "at": act["at"],
            "targets": dict(p["targets"]), "name": None,
            "frame": int(math.ceil(t * fps)), "ok": bool(p.get("ok", True))}))
    for m in mvs:
        act = acts[m["act"]]
        f0, f1 = int(m["start"] * fps), int(math.ceil(
            (m["start"] + m["seconds"]) * fps))
        mine = [q for q in set(m["add"]) | set(m["abs"]) if q in bm.params]
        best, bf = -1.0, f0
        for f in range(max(f0, 0), min(f1, n - 1) + 1):
            over = snaps[f][0]
            dev = sum(abs(over.get(q, rest[q]) - rest[q]) / span(q)
                      for q in mine)
            if dev > best:
                best, bf = dev, f
        body_keys.append((m["act"], {
            "kind": "move", "t": round(bf / fps, 3), "at": act["at"],
            "targets": {"name": m["name"]}, "name": m["name"],
            "frame": bf, "ok": True}))
    # A [frame] marker nothing follows (before the next frame marker or
    # the end) gets a still of its own.
    order = sorted(body_keys, key=lambda x: x[0])
    frame_acts = [(i, a) for i, a in enumerate(acts)
                  if a["tag"] == pz.FRAME and a.get("arg") in fr.FRAMES]
    for j, (i, a) in enumerate(frame_acts):
        nxt = frame_acts[j + 1][0] if j + 1 < len(frame_acts) else len(acts)
        if not any(i < ai < nxt for ai, _k in order) and framed:
            t = float(a["t"]) + fr.VIEW_EASE
            order.append((i, {"kind": "frame", "t": round(t, 3),
                              "at": a["at"], "targets": {"frame": a["arg"]},
                              "name": None, "frame": int(math.ceil(
                                  t * fps)), "ok": True}))
    order.sort(key=lambda x: x[0])
    for k, (_i, key) in enumerate(order, 1):
        over, parts = snap(key.pop("frame"))
        view, followed = view_at(key["t"])
        key.update(n=k, view=view, followed=followed, params=over,
                   parts=parts, words=_words_after(spoken, key["at"]))
        what = (f"move {key['name']}" if key["kind"] == "move"
                else f"frame {key['targets']['frame']}"
                if key["kind"] == "frame"
                else _targets_text(key["targets"]))
        key["what"] = what
        key["caption"] = [f"{k} {view}" + (" (follows)" if followed else "")
                          + ("" if key["ok"] else " MISS"),
                          f"\"{key['words']}\"" if key["words"] else ""]
        keys.append(key)
    return Walk(keys=keys, spoken=spoken, home=home, tags=tags,
                warnings=list(parser.warnings), start=start, end=end)


# ---------------------------------------------------------------- renderers

class FakeRenderer:
    """No GL: flow.live.FakeBackend's flat image, a stamp of the camera and
    the head turn on it. `views` are stand-in set_view states. calls keeps
    (view, params) of every render, for tests."""

    kind = "fake"

    def __init__(self, size=96):
        try:
            from .live import FakeBackend
        except ImportError:
            from flow.live import FakeBackend
        self.be = FakeBackend(size=size)
        self.views = {"bust": (1.8, 0.0, -0.4), "full": (1.0, 0.0, 0.0)}
        self.calls = []
        self.cur = None

    def render(self, params, parts, view):
        if view != self.cur:
            self.be.set_view(*self.views[view])
            self.cur = view
        self.calls.append((view, dict(params)))
        return self.be.render(params, parts=parts)

    def close(self):
        pass


class Live2DRenderer:
    """One Live2DBackend for an avatar, calibrated once, with its two
    camera states (framing.bust_view and the whole canvas); render() sets
    the view when it changes and draws the still. Needs GL (xvfb-run -a).
    squint: the eye smile that closes the eye (flow.squint) on a rig whose
    smile shows only on a closing eye, as the live and offline renders do."""

    kind = "live2d"

    def __init__(self, model_json, body_path, size=RENDER_SIZE, log=_quiet,
                 squint=True):
        try:
            from ..l2dprobe.backend import Live2DBackend
        except ImportError:
            from l2dprobe.backend import Live2DBackend
        self.be = Live2DBackend(model_json, size=size)
        self.be.__enter__()
        self.be.calibrate()
        full = (1.0, 0.0, 0.0)
        bust = fr.bust_view(self.be, body_path, model_json, log=log)
        if bust is None:
            log(f"judge: {model_json}: no bust view; bust renders full")
        self.views = {"bust": bust or full, "full": full}
        self.cur = None
        self.squint = sq.for_model(model_json, body_path) if squint else None

    def render(self, params, parts, view):
        if view != self.cur:
            self.be.set_view(*self.views[view])
            self.cur = view
        if self.squint:
            params = self.squint.apply(params)
        return self.be.render(params, parts=parts)

    def close(self):
        self.be.close()


def _find_file(path, name=None):
    for c in (path, os.path.join(ROOT, path or ""),
              os.path.join(ROOT, "rigs", os.path.basename(path or name or ""))):
        if c and os.path.isfile(c):
            return c
    return None


class Avatars:
    """The renderable avatars of a data folder: body model and renderer by
    avatar name, made on first use and kept (one backend per avatar).
    `renderer`: "live2d" or "fake". get(name) -> (bm, renderer or None,
    reason): renderer None, with the reason, for a VRM (not supported) and
    for a Live2D avatar whose model3.json is not on this machine."""

    def __init__(self, data, renderer="live2d", models=(), size=RENDER_SIZE,
                 log=_quiet, squint=True):
        self.reg = dx.load_registry(data).get("avatars", {}) if data else {}
        self.mode, self.size, self.log = renderer, size, log
        self.squint = squint
        self.roots = tuple(models) + MODEL_ROOTS
        self.made = {}

    def spec(self, name):
        return (self.reg.get(name) or {}).get("spec") or {}

    def get(self, name):
        if name not in self.made:
            self.made[name] = self._make(name)
        return self.made[name]

    def body_path(self, name):
        spec = self.spec(name)
        return _find_file(spec.get("body") or "", f"{name}.body.json") \
            or _find_file(os.path.join("rigs", f"{name}.body.json"))

    def body(self, name):
        """The avatar's body model (a VRM's too), or None; no renderer."""
        path = self.body_path(name)
        if path is None:
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                kind = json.load(fh).get("kind", "live2d")
            if kind == "vrm":
                try:
                    from . import vrm as fv
                except ImportError:
                    from flow import vrm as fv
                return fv.body_for(None, path, log=_quiet)[0]
            return fb.load(path)
        except Exception:  # noqa: BLE001 - an unreadable body: unknown
            return None

    def _make(self, name):
        spec = self.spec(name)
        path = self.body_path(name)
        if path is None:
            return None, None, f"no body file for {name}"
        with open(path, encoding="utf-8") as fh:
            kind = json.load(fh).get("kind", "live2d")
        if kind == "vrm":
            return None, None, "VRM is not supported (NA)"
        bm = fb.load(path)
        if self.mode == "fake":
            return bm, FakeRenderer(), None
        model = spec.get("model")
        if not model or not os.path.exists(model):
            model = ds._find_model(name, self.roots)
        if not model:
            return bm, None, f"no {name}.model3.json here (NA)"
        return bm, Live2DRenderer(model, path, self.size, self.log,
                                  self.squint), None

    def close(self):
        for _bm, r, _why in self.made.values():
            if r is not None:
                r.close()


# ------------------------------------------------------------- contact sheet

_FONTS = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
          "/mnt/c/Windows/Fonts/YuGothM.ttc", "/mnt/c/Windows/Fonts/msyh.ttc",
          "/mnt/c/Windows/Fonts/meiryo.ttc")


def _font(px):
    """(font, has CJK glyphs)."""
    from PIL import ImageFont
    for f in _FONTS:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, px), True
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=px), False
    except TypeError:                         # Pillow before 10.1
        return ImageFont.load_default(), False


def _ascii(text):
    return "".join(c if ord(c) < 0x250 else "?" for c in text)


def to_image(frame, size):
    """An RGBA / RGB array (or PIL image) as an RGB tile on the sheet's
    background, size x size."""
    from PIL import Image
    im = frame if hasattr(frame, "convert") else Image.fromarray(
        np.ascontiguousarray(frame))
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, BG + (255,))
    bg.alpha_composite(im)
    return bg.convert("RGB").resize((size, size), Image.LANCZOS)


def priority(key) -> int:
    """How much a still tells the judge: moves and hand poses most, a turn
    of the head or eyes next, the rest least."""
    if key["kind"] == "move":
        return 3
    t = key["targets"]
    if key["kind"] == "pose":
        if any(dx.is_hand(k) for k in t):
            return 3
        yaw, gx = t.get("head.yaw"), t.get("gaze.x")
        if (isinstance(yaw, (int, float)) and abs(yaw) >= 7) or (
                isinstance(gx, (int, float)) and abs(gx) >= 0.3):
            return 2
    return 1


def pick(keys, max_tiles=MAX_TILES) -> list:
    """The stills a sheet shows: the start state and the max_tiles - 1 most
    telling others (priority, then earlier), in order."""
    rest, others = keys[:1], keys[1:]
    top = sorted(others, key=lambda k: (-priority(k), k["n"]))[:max(
        max_tiles - 1, 0)]
    return rest + sorted(top, key=lambda k: k["n"])


def compose(tiles, size=TILE, lines=3, log=_quiet):
    """(PIL image, caption font had CJK): tiles = [(image, [caption
    lines])] left to right; each caption wrapped to the tile's width."""
    from PIL import Image, ImageDraw
    px = max(10, size // 22)
    font, cjk = _font(px)
    pad = 4
    cap_h = (lines + 1) * (px + 3) + pad
    sheet = Image.new("RGB", (size * len(tiles), size + cap_h), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    for i, (im, caption) in enumerate(tiles):
        x = i * size
        sheet.paste(im, (x, 0))
        d.rectangle([x, 0, x + size - 1, size + cap_h - 1], outline=(120,) * 3)
        y, used = size + pad, 0
        for line in caption:
            line = line if cjk else _ascii(line)
            while line and used <= lines:
                n = len(line)
                while n > 1 and d.textlength(line[:n], font=font) \
                        > size - 2 * pad:
                    n -= 1
                d.text((x + pad, y), line[:n], fill=(10, 10, 10), font=font)
                y += px + 3
                used += 1
                line = line[n:].lstrip()
    return sheet, cjk


class Sheet:
    """png: the contact sheet; keys: the shown stills without images
    (kind, n, view, words, targets ...); n_keys: all stills of the reply."""

    def __init__(self, png, keys, n_keys, walk_, avatar=None, note=None):
        self.png, self.keys, self.n_keys = png, keys, n_keys
        self.walk, self.avatar, self.note = walk_, avatar, note

    @property
    def sha(self):
        return hashlib.sha256(self.png).hexdigest()


def _png(img, keys):
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("tsk.keys", json.dumps(keys, ensure_ascii=False))
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def _meta(key):
    return {k: key[k] for k in ("kind", "n", "view", "words", "targets",
                                "name", "followed") if k in key}


def render_sheet(bm, renderer, raw, home="bust", now=None, seed=0,
                 max_tiles=MAX_TILES, tile=TILE, follow=True,
                 avatar=None) -> Sheet:
    """walk() the reply, render the stills the sheet shows (the renderer is
    drawn on serially: GL stays on one thread) and compose them."""
    w = walk(bm, raw, home, now, seed, follow)
    shown = pick(w["keys"], max_tiles)
    tiles = []
    for key in shown:
        img = to_image(renderer.render(key["params"], key["parts"],
                                       key["view"]), tile)
        tiles.append((img, key["caption"]
                       + ([key["what"]] if SHOW_TARGETS and key.get("what")
                          else [])))
    img, _cjk = compose(tiles, tile)
    meta = [_meta(k) for k in shown]
    return Sheet(_png(img, meta), meta, len(w["keys"]) - 1, w, avatar)


# ------------------------------------------------------------------ the judge

RUBRIC = """\
You judge an animated streamer character by looking at it as the viewer \
does. You get ONE image: a contact sheet of still frames of the character, \
left to right in the order the reply plays. The first tile is the start \
state (rest) for reference. Each other tile is the character at the peak of \
one body marker, captioned with its number, its camera ("bust": head and \
chest only; "full": the whole body) and the words it comes before. You also get the viewer's message, the words the character says \
and what its body can do.

Sides: "your right" / "your left" in a request mean the CHARACTER's own \
right and left. The character faces the viewer like a mirror: its right is \
on the VIEWER's LEFT in the picture. Turning toward its right is the head \
and eyes moving toward the picture's left.

Score:
- request_done (0-2): did the viewer ask the character for a body action \
(point, wave, raise a hand, a pose, look or turn left / right / up / down)? \
2 = visibly done in the frames, by the right part (a hand request needs the \
hand; a look needs the head or eyes), toward the right side; 1 = partly (the \
right action but weak, or the part or side unclear); 0 = not done, the wrong \
side, or only the face moved. null (not applicable) when no body action was \
asked.
- visible (0-2): are the body moves of the sheet (hands, arms, body) inside \
the picture of their camera and readable? 2 = clearly visible; 1 = partly \
cropped or tiny; 0 = out of the picture (a hand below the bottom edge of a \
bust picture is not seen) or no visible change. null when the reply makes \
no hand, arm or body move.
- expression_fit (0-2): does the face match what is being said? 2 = fits; \
1 = neutral or weak; 0 = clashes (a grin for bad news, a frown for thanks).
- natural (1-5): does it look like a natural streamer? 5 = lively, varied \
and relaxed; 3 = acceptable; 1 = stiff, the same pose again and again, \
jerky or extreme.
- issues: up to four short strings naming concrete problems; [] if none.

Judge only what the frames show; do not give credit for what the words \
claim. Reply with ONE JSON object and nothing else:
{"request_done": 0|1|2|null, "visible": 0|1|2|null, "expression_fit": \
0|1|2, "natural": 1|2|3|4|5, "issues": ["..."]}"""


def budget_text(b) -> str:
    """The body's motion budget in a sentence (framing.budget's dict from
    avatars.json): whether and how far its arms move."""
    if not b:
        return "unknown"
    hm = b.get("hand_moves") or {}
    reach = (b.get("reach") or {}).get("all")
    if not hm.get("full", 1):
        arms = "its arms do not move on their own (no hand action possible)"
    else:
        arms = ("its arms move" + (f", a hand reaches about {reach:.0%} of "
                                   "the places a hand goes when talking"
                                   if isinstance(reach, (int, float)) else ""))
        arms += (f"; hand moves a reply: bust {hm.get('bust', '?')}, "
                 f"full {hm.get('full', '?')}")
    return f"{arms}; {b.get('face', '?')} face controls"


def prompt_text(user, spoken, budget, keys, avatar=None,
                show_targets=None) -> str:
    """The text part of the judge's request; the markers' targets only with
    show_targets (SHOW_TARGETS by default)."""
    show = SHOW_TARGETS if show_targets is None else show_targets
    rows = []
    for i, k in enumerate(keys, 1):
        if k["kind"] == "rest":
            rows.append(f"tile {i}: start state, camera {k['view']}")
        else:
            what = (f"move {k['name']}" if k["kind"] == "move"
                    else f"frame {k['targets'].get('frame')}"
                    if k["kind"] == "frame" else _targets_text(k["targets"]))
            rows.append(f"tile {i}: camera {k['view']}"
                        + (" (the camera followed a whole-body move)"
                           if k.get("followed") else "")
                        + (f", before \"{k['words']}\"" if k["words"] else "")
                        + (f", {what}" if show else ""))
    return (f"Viewer's message (with the lines the system adds):\n{user}\n\n"
            f"The character says (markers removed):\n{spoken or '(nothing)'}"
            f"\n\nThe character{f' ({avatar})' if avatar else ''}: "
            f"{budget_text(budget)}.\n\nTiles:\n" + "\n".join(rows)
            + "\n\nJSON only.")


FIELDS = ("request_done", "visible", "expression_fit", "natural")
RANGES = {"request_done": (0, 2), "visible": (0, 2), "expression_fit": (0, 2),
          "natural": (1, 5)}


def _num(v, lo, hi):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "na", "n/a", "null", "none", "-"):
            return None
        try:
            v = float(s)
        except ValueError:
            raise ValueError(v)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) \
            or not math.isfinite(v):
        raise ValueError(v)
    return int(min(max(round(v), lo), hi))


def parse_judgment(text):
    """The judge's scores from its reply: the first JSON object in the text
    (a code fence, a <think> block or prose around it are ignored), each
    score clamped to its range, "NA" / null as None. {request_done,
    visible, expression_fit, natural, issues} or None when there is no
    usable object (expression_fit and natural are required)."""
    s = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s[m.start():])
        except ValueError:
            continue
        if not isinstance(obj, dict) or not (
                "expression_fit" in obj or "natural" in obj):
            continue
        try:
            out = {f: _num(obj.get(f), *RANGES[f]) for f in FIELDS}
        except ValueError:
            continue
        if out["expression_fit"] is None or out["natural"] is None:
            continue
        issues = obj.get("issues")
        out["issues"] = [str(x)[:200] for x in issues[:4]] if isinstance(
            issues, list) else ([str(issues)[:200]] if issues else [])
        return out
    return None


# ------------------------------------------------------------ judge backends

JUDGE_MODELS = {"siliconflow": "Qwen/Qwen2.5-VL-72B-Instruct",
                "vllm": "Qwen/Qwen3-VL-8B-Instruct",
                "anthropic": dr.DEFAULT_MODEL["anthropic"],
                "fake": "fake"}
JUDGE_BACKENDS = tuple(JUDGE_MODELS)


class FakeJudgeBackend:
    """A deterministic stand-in for tests and dry runs: reads the sheet's
    own metadata (the shown stills, PNG text "tsk.keys") and the viewer's
    message from the request and scores what a careful judge would: a
    look to the asked side, a hand marker for a hand request, a hand in
    bust shown only above the window. `fn(png, text, keys) -> dict`
    replaces it. calls counts requests."""

    def __init__(self, fn=None, raw=None):
        self.fn, self.raw = fn, raw
        self.model, self.stop_reason, self.usage = "fake", "stop", {}
        self.calls = 0

    @staticmethod
    def decode(messages):
        c = messages[0]["content"]
        png, text = b"", ""
        for blk in c if isinstance(c, list) else []:
            if blk.get("type") == "image_url":
                png = base64.b64decode(blk["image_url"]["url"].split(",", 1)[1])
            elif blk.get("type") == "text":
                text = blk["text"]
        return png, text

    def stream(self, system, messages):
        from PIL import Image
        self.calls += 1
        png, text = self.decode(messages)
        keys = []
        if png:
            keys = json.loads(Image.open(io.BytesIO(png)).info.get(
                "tsk.keys", "[]"))
        self.usage = {"prompt_tokens": 800 + len(text) // 4,
                      "completion_tokens": 40}
        if self.raw is not None:
            out = self.raw(self.calls) if callable(self.raw) else self.raw
        else:
            out = json.dumps((self.fn or fake_scores)(png, text, keys))
        yield out


def fake_scores(png, text, keys) -> dict:
    """FakeJudgeBackend's default scores from the shown stills."""
    try:
        from . import checks as ck
    except ImportError:
        from flow import checks as ck
    user = text.split("Viewer's message", 1)[-1].split(
        "The character says", 1)[0]
    body = [k for k in keys if k["kind"] != "rest"]
    side = ck.asked_side(user)
    hand_ask = ha.asked_hand_action(user) or ha.asked_wave_action(user)
    hands = [k for k in body if k["kind"] == "move" or any(
        dx.is_hand(t) for t in k["targets"])]
    done = None
    if side is not None:
        want = -1 if side == "R" else 1
        done = 0
        for k in body:
            t = k["targets"]
            for v in (t.get("head.yaw"), t.get("gaze.x")):
                if isinstance(v, (int, float)) and abs(v) > 0 and (
                        (v > 0) - (v < 0)) == want:
                    done = 2
    elif hand_ask:
        done = 2 if hands else 0
    vis = None
    if hands:
        vis = 2
        for k in hands:
            if k["view"] == "bust" and k["kind"] == "pose":
                ys = [v for t, v in k["targets"].items()
                      if t.endswith(".y") and isinstance(v, (int, float))]
                if not ys or max(ys) < -1.0:
                    vis = 0
    dup = len({json.dumps(k["targets"], sort_keys=True) for k in body}) < len(
        body)
    return {"request_done": done, "visible": vis, "expression_fit": 1,
            "natural": 2 if dup else 4,
            "issues": (["the same beat twice"] if dup else [])}


class BudgetExceeded(RuntimeError):
    """--max-calls reached."""


class Judge:
    """Scores a sheet with a VLM. `factory()` makes a backend (one per
    call: usage and stop reason live on the instance, so threads do not
    share one); `fmt` the image content format ("anthropic" or
    "siliconflow" for the OpenAI-style data URL). Results are cached in
    `cache` (a folder) by (sheet sha, prompt sha, model, backend); a hit
    costs no call. `max_calls`: model calls allowed (retries count);
    BudgetExceeded past it. Counters: calls, cached, retries, failed (calls
    that raised: HTTP errors, timeouts, an empty stream), bad (answers
    never parsed), tokens_in, tokens_out; errors: {message: count} of the
    failures (status and the first 300 characters of the body)."""

    def __init__(self, factory, model, name="judge", fmt="siliconflow",
                 cache=None, max_calls=None, price=(None, None), log=_quiet):
        self.factory, self.model, self.name, self.fmt = factory, model, name, fmt
        self.cache, self.max_calls, self.price, self.log = (
            cache, max_calls, price, log)
        self.calls = self.cached = self.retries = self.bad = 0
        self.failed = 0
        self.errors = {}
        self.tokens_in = self.tokens_out = 0
        self.lock = threading.Lock()

    def _key(self, sheet_sha, text):
        h = hashlib.sha256()
        for part in (RUBRIC, text, self.model, self.name):
            h.update(part.encode("utf-8"))
            h.update(b"\0")
        return sheet_sha[:24] + "-" + h.hexdigest()[:24]

    def _path(self, key):
        return os.path.join(self.cache, key[:2], key + ".json")

    def _call(self, messages):
        with self.lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded(f"--max-calls {self.max_calls} reached")
            self.calls += 1
        be = self.factory()
        try:
            text = "".join(be.stream(RUBRIC, messages))
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 - counted, shown, re-raised
            self._failure(f"{type(exc).__name__}: {exc}")
            raise
        u = getattr(be, "usage", None) or {}
        if not text.strip():
            # A 200 with nothing in it (a reasoning model that spent its
            # tokens thinking, a filter): shown, then counted as unparsed.
            self._failure(f"empty answer (stop {getattr(be, 'stop_reason', None)}"
                          f", usage {u})", count=False)
        with self.lock:
            self.tokens_in += int(u.get("prompt_tokens")
                                  or u.get("input_tokens") or 0)
            self.tokens_out += int(u.get("completion_tokens")
                                   or u.get("output_tokens") or 0)
        return text

    def _failure(self, msg, count=True):
        """Record a failed call and print it the first time (stderr): the
        provider's status and the first 300 characters of its body."""
        msg = msg.replace("\n", " ")[:300]
        with self.lock:
            if count:
                self.failed += 1
            first = msg not in self.errors
            self.errors[msg] = self.errors.get(msg, 0) + 1
            shown = len(self.errors) <= 5
        if first and shown:
            print(f"judge {self.name}/{self.model}: call failed: {msg}",
                  file=sys.stderr)

    def score(self, sheet, user, spoken, budget=None, avatar=None) -> dict:
        """{scores: parsed judgment or None, raw, cached, key}."""
        text = prompt_text(user, spoken, budget, sheet.keys, avatar)
        key = self._key(sheet.sha, text)
        if self.cache and os.path.exists(self._path(key)):
            try:
                with open(self._path(key), encoding="utf-8") as fh:
                    hit = json.load(fh)
                with self.lock:
                    self.cached += 1
                return dict(hit, cached=True)
            except (OSError, ValueError):
                pass
        msg = pz.user_message(text, self.fmt, sheet.png)
        raw = self._call([msg])
        scores = parse_judgment(raw)
        if scores is None:
            with self.lock:
                self.retries += 1
            again = [msg, {"role": "assistant", "content": raw or "(empty)"},
                     {"role": "user", "content": "That was not valid. Reply "
                      "with the one JSON object only, as specified."}]
            raw = self._call(again)
            scores = parse_judgment(raw)
            if scores is None:
                with self.lock:
                    self.bad += 1
        out = {"scores": scores, "raw": raw, "cached": False, "key": key}
        if scores is not None and self.cache:
            os.makedirs(os.path.dirname(self._path(key)), exist_ok=True)
            tmp = self._path(key) + f".{os.getpid()}.{threading.get_ident()}"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"scores": scores, "raw": raw, "key": key,
                           "model": self.model}, fh, ensure_ascii=False)
            os.replace(tmp, self._path(key))
        return out

    def report(self) -> str:
        pin, pout = self.price
        cost = (f", about ${(self.tokens_in * pin + self.tokens_out * pout) / 1e6:.3f}"
                if pin is not None and pout is not None else "")
        errs = "".join(f"\n  {n}x {m}" for m, n in sorted(
            self.errors.items(), key=lambda x: -x[1])[:3])
        return (f"judge {self.name}/{self.model}: {self.calls} calls "
                f"({self.retries} retries, {self.failed} failed, "
                f"{self.bad} unparsed), "
                f"{self.cached} from cache, tokens {self.tokens_in} in / "
                f"{self.tokens_out} out{cost}{errs}")


def make_judge(backend, model=None, api_base=None, api_key_env=None,
               cache=None, max_calls=None, price=(None, None),
               fake=None, log=_quiet) -> Judge:
    """A Judge on a backend: siliconflow / vllm (director.OpenAICompatBackend
    with temperature 0; images as base64 data URLs), anthropic
    (director.AnthropicBackend; no temperature control), fake."""
    if backend not in JUDGE_BACKENDS:
        raise SystemExit(f"--judge-backend {backend!r}: one of "
                         f"{', '.join(JUDGE_BACKENDS)}")
    model = model or JUDGE_MODELS[backend]
    if backend == "fake":
        be = fake or FakeJudgeBackend()
        return Judge(lambda: be, model, backend, "siliconflow", cache,
                     max_calls, price, log)
    if backend == "anthropic":
        return Judge(lambda: dr.AnthropicBackend(model, "low", 512, log=log),
                     model, backend, "anthropic", cache, max_calls, price, log)
    return Judge(lambda: dr.OpenAICompatBackend(
        backend, model, api_base, api_key_env, max_tokens=512, timeout=180,
        log=log, sampling={"temperature": 0}), model, backend, "siliconflow",
        cache, max_calls, price, log)


# ------------------------------------------------------------------------ cli

def add_judge_args(ap):
    ap.add_argument("--judge-backend", default="fake", choices=JUDGE_BACKENDS,
                    help="the VLM: siliconflow (OpenAI-compatible, needs "
                         "SILICONFLOW_API_KEY), vllm (a local server, "
                         "--api-base [http://localhost:8000/v1]), "
                         "anthropic, fake [fake]")
    ap.add_argument("--judge-model", default=None,
                    help="model name [siliconflow: "
                         f"{JUDGE_MODELS['siliconflow']}; vllm: "
                         f"{JUDGE_MODELS['vllm']}]; check the provider's "
                         "model list")
    ap.add_argument("--judge-api-base", default=None)
    ap.add_argument("--judge-key-env", default=None)
    ap.add_argument("--judge-cache", default=None,
                    help="folder for cached judgments [none]")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="stop after this many model calls (cache hits "
                         "are free)")
    ap.add_argument("--price", nargs=2, type=float, default=None,
                    metavar=("IN", "OUT"),
                    help="US$ per million input / output tokens, to print "
                         "a cost")


def add_render_args(ap):
    ap.add_argument("--renderer", default="live2d", choices=("live2d", "fake"),
                    help="live2d (GL: xvfb-run -a) or fake [live2d]")
    ap.add_argument("--models", action="append", default=[],
                    help="a folder to find <avatar>.model3.json under "
                         "(repeatable); also live2d-models, CubismWebSamples")
    ap.add_argument("--tiles", type=int, default=MAX_TILES,
                    help=f"stills on a sheet, the start state included "
                         f"[{MAX_TILES}]")
    ap.add_argument("--tile", type=int, default=TILE,
                    help=f"tile size in px [{TILE}]")
    sq.add_args(ap)
    ap.add_argument("--no-follow", dest="follow", action="store_false",
                    help="the camera does not follow whole-body moves")
    ap.add_argument("--show-targets", action="store_true",
                    help="show the judge the markers' targets under the "
                         "tiles and in the tile list [off: it judges the "
                         "frames]")


def judge_from_args(a, log=print) -> Judge:
    global SHOW_TARGETS
    SHOW_TARGETS = bool(getattr(a, "show_targets", False))
    return make_judge(a.judge_backend, a.judge_model, a.judge_api_base,
                      a.judge_key_env, a.judge_cache, a.max_calls,
                      tuple(a.price) if a.price else (None, None), log=log)


def find_record(data, rid):
    for r in ds.iter_records(data):
        if r["id"] == rid:
            return r
    raise SystemExit(f"{rid}: not in {data}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "score"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="a flow.dataset folder")
        p.add_argument("--id", required=True, help="SESSION/tNN")
        p.add_argument("--reply", default=None,
                       help="the raw reply [the record's reply_raw]")
        p.add_argument("--reply-file", default=None)
        p.add_argument("--out", default=None, help="sheet PNG to write")
        add_render_args(p)
        if name == "score":
            add_judge_args(p)
    a = ap.parse_args(argv)
    global SHOW_TARGETS
    SHOW_TARGETS = bool(getattr(a, "show_targets", False))
    if a.renderer == "live2d":
        try:
            from ..l2dprobe.backend import ensure_display
        except ImportError:
            from l2dprobe.backend import ensure_display
        ensure_display()
    r = find_record(a.data, a.id)
    raw = a.reply
    if a.reply_file:
        with open(a.reply_file, encoding="utf-8") as fh:
            raw = fh.read()
    raw = r["reply_raw"] if raw is None else raw
    av = Avatars(a.data, a.renderer, a.models,
                 squint=not a.no_smile_squint)
    name = r["avatar"]["name"]
    bm, rend, why = av.get(name)
    if rend is None:
        print(f"judge: {name}: {why}")
        return 2
    try:
        sheet = render_sheet(bm, rend, raw, home_of(r),
                             now_targets(r["user_content"]), r.get("seed", 0),
                             a.tiles, a.tile, a.follow, name)
    finally:
        av.close()
    if a.out:
        with open(a.out, "wb") as fh:
            fh.write(sheet.png)
        print(f"sheet -> {a.out} ({len(sheet.keys)} tiles of "
              f"{sheet.n_keys} stills)")
    if a.cmd == "score":
        j = judge_from_args(a)
        budget = dx.load_registry(a.data)["avatars"].get(name, {}).get(
            "budget")
        got = j.score(sheet, r["user_content"], dx.spoken_of(raw), budget,
                      name)
        print(json.dumps(got["scores"], ensure_ascii=False, indent=1))
        print(j.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
