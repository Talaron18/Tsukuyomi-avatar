#!/usr/bin/env python3
"""
Failure checkers for a student's replies: the failures seen live and in the
held-out evals as numbers per model, to compare models side by side and to
find the turns a preference tuning should push on. No model is called.

Input (load()): an evalsft `student.jsonl` (or the folder holding it, with
the prompt conditions `student.full+rem.jsonl` / `student.short+rem.jsonl`
(`evalsft run --prompt-cond`) and `recorded.jsonl` as further inputs when
there are any; the labels are "eval/student", "eval/student.full+rem" ...
and a folder's are "eval:student", "eval:student.full+rem"), flow.dataset shards
(`data/.../shards/*.jsonl`, or the folder), or a flow.live `--record`
turns.jsonl (lines, raw replies, `quiet`, `framing`; no avatar: it comes
from --avatar or the folder's name). Per record: user line, raw reply,
spoken text, avatar, session, turn.

Every check gives each record "pass", "fail" or "na" (the check does not
apply to it) and an aggregate: counts, the fail rate among the applicable
records (lower is better for all of them), and per avatar the same.

  look_sign          The user line asks to look or turn to your right or
                     left (en "your right" / "to the left" with look, turn,
                     glance; ja 右 / 左 with 見 / 向; zh 右 / 左 with 看 / 转
                     / 轉, 右邊 too): the first pose marker with a head.yaw
                     of at least 7 degrees or a gaze.x of at least 0.3 must
                     have the sign of pose's convention (head.yaw and gaze.x:
                     + toward your left), so right is negative. A line
                     naming both sides, or none, is na. A reply that turns
                     nowhere is a fail too, told apart in the aggregate
                     (wrong_sign / no_turn).
  glance_at_mention  The user line puts something near her without asking
                     her to look (en "at your right foot", "behind you",
                     "on your left"; ja 足元, 右足に, 後ろに, 左側に置いた; zh
                     你脚边, 你右边, 身后; not "my right hand hurts", "turn
                     right at the station", "on my right"): the first or
                     second pose marker must turn her head or gaze toward
                     it (her right is negative; feet: down). Fail told
                     apart in the aggregate (no_pose / no_glance).
  look_at_me         The user line asks her to look at me / the camera / the
                     chat (en "look at me", "look at the camera", "face me";
                     ja こっち見て, 私を見て, カメラ見て; zh 看着我, 看我, 看镜头;
                     not "don't look at me", not a line that also names a
                     side, which look_sign reads): the head.yaw and gaze.x
                     the reply ends on (the last pose marker that sets each)
                     must not be clearly sideways: |head.yaw| < 7 degrees
                     and |gaze.x| < 0.3. A reply that sets neither passes
                     (nothing turns her away). Fail told apart in the
                     aggregate (yaw / gaze / both).
  correction_obeyed  The user corrects a word of hers ("not X but Y", "it's Y,
                     not X", "you said X, it's Y"; ja XじゃなくてY, Yだ、X
                     じゃない; zh 不是X，是Y, 是Y不是X; or a "no / 違う / 不对"
                     line with a word close to one in her last reply) and
                     the reply still says X. na without a correction.
  pose_copy          The reply's pose markers (variables and rounded values,
                     dur / hold aside) equal, or at least 3 of 4 of its beats
                     equal in place, those of one of her previous 3 replies
                     in the session (the collapse seen live, a generalised
                     template_repeat). na for a reply with fewer than 2 poses
                     and for the first reply of a session.
  wave_obeyed        The line asks to wave (en wave, ja 手を振 / バイバイ, zh
                     挥手 / 揮手): the reply has a wave move or a hand target.
                     na for a body whose hands reach nothing (the motion
                     budget, flow.framing.budget: hand_moves full 0).
  body_request_obeyed  The line explicitly asks the character for a hand or
                     arm action (flow.handask: point, wave, raise a hand,
                     peace sign, thumbs up, clap, hands on hips, cross
                     arms, a heart ...; en / ja / zh): at least one hand
                     target, move or gesture. Not counted: the topic (an
                     invitation, a tease), a request for the head or the
                     eyes alone (look, turn, nod), a mention in the past or
                     a compliment. na for a body without hand reach.
  opening            The first pose marker's variable set (and rounded
                     values) over the replies: fail marks the q8r2
                     template (exactly brow.L.lift, brow.R.lift, head.pitch,
                     head.yaw with brows 0.6-0.9, pitch 4-6, yaw -4 / -5);
                     the aggregate also has the top opening's share (by
                     variable set, and with rounded values).
  template_repeat    Within a session a reply whose sequence of markers (the
                     variable set of each pose, the tag of the others) is
                     the one of the reply before: the collapse seen live.
                     na for the first reply of a session and for replies
                     without a marker.
  language           The reply language (flow.pose.detect_lang of the spoken
                     text) differs from the user line's. na for mixed lines,
                     quiet turns and undetected languages.
  frame_for_dance    A whole-body move (flow.framing.WHOLE_BODY) in a reply
                     that starts in bust without a [frame full] before it.
                     Informational: the runtime's camera follows anyway. na
                     without such a move, for a body that never switches
                     and for a reply that starts in full.
  frame_before_wave  A [frame full] before a wave (should be about 0: a wave
                     stays in bust). na without a wave move.
  beats              Beats (flow.dataexport beat_counts) per sentence and per
                     reply; fail: fewer beats than sentences. na without
                     speech.
  quiet_says_quiet   A quiet-chat turn (the line flow.pose.QUIET_CHAT)
                     whose spoken text mentions the chat being quiet (en
                     "chat is quiet", "no one is talking"; ja チャット ...
                     静か; zh 没人说话 / 聊天室安静). Informational.

Under the checks the table also shows the style numbers of flow.style (words
and sentences a reply, distinct-n, echo of the user line, question endings,
openings, interjections, a substance proxy): numbers, not pass / fail, and not
in CHECKS. Definitions in flow/style.py and docs/embodiment.md ("Style
checks").

Usage:
    python scripts/checks.py runs/eval_q8m/student.jsonl \\
        runs/eval_q8r2/student.jsonl [--by avatar] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from . import body as fb
    from . import dataexport as dx
    from . import dataset as ds
    from . import framing as fr
    from . import handask as ha
    from . import pose as pz
    from . import style as sty
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from flow import body as fb
    from flow import dataexport as dx
    from flow import dataset as ds
    from flow import framing as fr
    from flow import handask as ha
    from flow import pose as pz
    from flow import style as sty

PASS, FAIL, NA = "pass", "fail", "na"
# A turn of the head or the eyes counts as a look: degrees, gaze units.
YAW_MIN = 7.0
GAZE_MIN = 0.3
# The q8r2 opening: the keys, and the ranges of their values.
Q8R2_KEYS = frozenset(("brow.L.lift", "brow.R.lift", "head.pitch",
                       "head.yaw"))
Q8R2_RANGE = {"brow.L.lift": (0.6, 0.9), "brow.R.lift": (0.6, 0.9),
              "head.pitch": (4.0, 6.0), "head.yaw": (-5.0, -4.0)}
FIDS = 50           # failing ids kept per check in the json
# The "kind" of the turns.jsonl lines that hold no reply.
NOT_REPLIES = ("memory", "repair")


# --------------------------------------------------------------- records

def _num(x):
    return float(x) if isinstance(x, (int, float)) \
        and not isinstance(x, bool) else None


class Rec(dict):
    """One reply: id, avatar, session, turn, user, raw, spoken, quiet,
    start (the framing the reply starts in, or None), target (the length
    the teacher was asked for, a dataset record's length_target, or None),
    markers (the pose parser's tags, parsed on first use)."""

    @property
    def markers(self) -> list:
        if "_markers" not in self:
            self["_markers"] = [t for _a, _b, t in
                                dx.marker_spans(self["raw"] or "")]
        return self["_markers"]


def make_rec(raw, user, avatar, session, turn, rid=None, spoken=None,
             quiet=None, start=None, topic=None, mix=None,
             target=None) -> Rec:
    """A record from its fields (the loaders and the tests build them)."""
    raw = raw or ""
    if quiet is None:
        quiet = (user or "").strip() == pz.QUIET_CHAT
    return Rec(id=rid or f"{session}/t{int(turn):02d}", avatar=avatar,
               session=session, turn=int(turn), user=user or "", raw=raw,
               spoken=dx.spoken_of(raw) if spoken is None else spoken,
               quiet=bool(quiet), start=start, topic=topic, mix=mix,
               target=target)


def _from_row(r, default_avatar, default_session) -> Rec:
    """A record from a line of a shard, an evalsft jsonl or a live
    turns.jsonl."""
    av = r.get("avatar")
    if isinstance(av, dict):
        av = av.get("name")
    raw = r.get("reply_raw") if "reply_raw" in r else r.get("reply")
    fr_ = r.get("framing")
    start = fr_.get("start") if isinstance(fr_, dict) else (
        fr_ if isinstance(fr_, str) else None)
    lang = r.get("lang") if isinstance(r.get("lang"), dict) else {}
    session = r.get("session") or default_session
    return make_rec(raw, r.get("user"), av or default_avatar, session,
                    r.get("turn") or 0, r.get("id"), r.get("spoken"),
                    r.get("quiet") or None, start, r.get("topic"),
                    lang.get("mix"), r.get("length_target"))


def _read(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def load(path, avatar=None) -> list:
    """[(label, [Rec])] for a path: an evalsft folder or file, a dataset
    folder or one of its shards, a live turns.jsonl (or its folder). A
    folder with student.jsonl and recorded.jsonl is two inputs."""
    base = os.path.basename(os.path.normpath(path))
    if os.path.isdir(path):
        if os.path.isdir(os.path.join(path, "shards")):
            recs = [_from_row(r, avatar, base)
                    for r in ds.iter_records(path)]
            return [(base, recs)]
        # student.jsonl, then the prompt conditions of flow.train.evalsft
        # (student.full+rem.jsonl, student.short+rem.jsonl), recorded.jsonl.
        names = ["student"] + sorted(
            f[:-len(".jsonl")] for f in os.listdir(path)
            if re.fullmatch(r"student\..+\.jsonl", f)) + ["recorded"]
        found = [(f"{base}:{n}", os.path.join(path, f"{n}.jsonl"))
                 for n in names
                 if os.path.exists(os.path.join(path, f"{n}.jsonl"))]
        if not found and os.path.exists(os.path.join(path, "turns.jsonl")):
            found = [(base, os.path.join(path, "turns.jsonl"))]
        if not found:
            raise SystemExit(f"{path}: no shards/, student.jsonl or "
                             "turns.jsonl")
        out = []
        for label, f in found:
            (_l, recs), = load(f, avatar or (base if "turns.jsonl" in f
                                             else None))
            out.append((label, recs))
        return out
    # A live turns.jsonl also holds flow.memory's consolidation records and
    # flow.repair's {"kind": "repair"} lines: they are not replies.
    rows = [r for r in _read(path) if r.get("kind") not in NOT_REPLIES]
    if not rows:
        raise SystemExit(f"{path}: no records")
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    # An evalsft file is labelled with its folder and its condition
    # ("eval/student", "eval/student.full+rem"): the same name in two runs
    # stays apart, and the conditions of one run side by side.
    label = f"{folder}/{stem}" if stem.split(".")[0] in (
        "student", "recorded", "turns") else stem
    if "reply_raw" not in rows[0] and "reply" not in rows[0]:
        raise SystemExit(f"{path}: not an evalsft, dataset or live record "
                         "file (no reply_raw / reply)")
    live = "reply_raw" not in rows[0]
    # A shard line is deduped by id like iter_records (the last wins).
    if not live and isinstance(rows[0].get("avatar"), dict):
        rows = list({r["id"]: r for r in rows}.values())
    recs = [_from_row(r, avatar or (folder if live else None),
                      folder + "/" + stem if live else stem)
            for r in rows]
    for r in recs:
        if r["avatar"] is None:
            r["avatar"] = "?"
    return [(label, recs)]


# ------------------------------------------------------------ the bodies

class Bodies:
    """What the checks need of an avatar's body, by its name: the motion
    budget (flow.framing.budget): from the data folder's avatars.json when
    given, else the body file rigs/NAME.body.json, else unknown (None: a
    body with hands that can switch the camera is assumed)."""

    def __init__(self, data=None, rigs="rigs", budgets=None):
        self.rigs, self.budgets = rigs, dict(budgets or {})
        if data:
            try:
                for k, v in (dx.load_registry(data).get("avatars")
                             or {}).items():
                    if v.get("budget"):
                        self.budgets.setdefault(k, v["budget"])
            except (OSError, ValueError):
                pass

    def budget(self, avatar):
        if avatar not in self.budgets:
            path = os.path.join(self.rigs, f"{avatar}.body.json")
            try:
                self.budgets[avatar] = fr.budget(fb.load(path)) \
                    if os.path.exists(path) else None
            except Exception:  # noqa: BLE001 - an unreadable body: unknown
                self.budgets[avatar] = None
        return self.budgets[avatar]

    def has_hands(self, avatar) -> bool:
        b = self.budget(avatar)
        if b is None:
            return True
        return bool((b.get("hand_moves") or {}).get("full", 1))

    def can_frame(self, avatar) -> bool:
        b = self.budget(avatar)
        return True if b is None else "bust" in (b.get("frames") or ())


# ----------------------------------------------------- the line's request

def _words(patterns, text) -> list:
    return [m for p in patterns for m in re.finditer(p, text, re.I)]


_PART = (r"hand|arm|elbow|shoulder|eye|cheek|ear|foot|leg|knee|wrist|"
         r"finger|brow")
_EN_SIDE = re.compile(r"\b(?:your|the)\s+(right|left)\b(?!\s+(?:" + _PART
                      + r")\b)|\bto\s+(?:the\s+)?(right|left)\b|\b(right|"
                      r"left)\s+(?:side|way)\b|\b(?:look|turn|glance|face)\w*\s+(right|"
                      r"left)\b", re.I)
_EN_LOOK = re.compile(r"\b(?:look|turn|glance|face|gaze|peek|stare|"
                      r"looking|turning|glancing|facing)\b", re.I)
# ja 見 / 向, zh 看 / 转 / 轉 / 向 / 望 within a few characters of 右 / 左
# (右邊 / 左邊 are the same characters); not 右手, 左眼, ... (a body part).
_SIDE = r"[右左](?![手眼腕肩耳脚腳腿足拳])"
_VERB = r"[見向看转轉望瞧]"
_CJK_SIDE = re.compile(f"({_VERB}.{{0,5}}?)({_SIDE})|({_SIDE}).{{0,5}}?"
                       f"{_VERB}")
# The character's own side named without a verb: ja そっち / 君の / あなたの
# + 右側, zh 你的 + 右边 / 左邊 (the line is a look request when it says
# "your side" this way; a bare 右 / 左 or a body part is not).
_ADDRESSEE = (r"(?:そっち|こっち|あっち|君|きみ|キミ|あなた|貴方|お前|おまえ|"
              r"你|您|妳)")
_CJK_YOUR = re.compile(_ADDRESSEE + r"(?:の|的)?(?:ほう|方)?(?:の|的)?"
                       r"([右左])(?:側|侧|边|邊|のほう|の方|方)"
                       r"(?![にでがはもと])")
_SIDES = {"right": "R", "left": "L", "右": "R", "左": "L"}


def asked_side(line):
    """"R", "L" or None: the side of a request to look or turn. None when
    the line asks no such thing or names both sides."""
    s = line or ""
    sides = set()
    for m in _EN_SIDE.finditer(s):
        near = s[max(0, m.start() - 40):m.end() + 15]
        if _EN_LOOK.search(near):
            sides.add(_SIDES[(m.group(1) or m.group(2) or m.group(3)
                              or m.group(4)).lower()])
    for m in _CJK_SIDE.finditer(s):
        sides.add(_SIDES[m.group(2) or m.group(3)])
    for m in _CJK_YOUR.finditer(s):
        sides.add(_SIDES[m.group(1)])
    return next(iter(sides)) if len(sides) == 1 else None


# "Look at me" / at the camera / at the chat: the character should end up
# facing the viewer, not a side. Imperative forms only ("are you looking at
# me" and "don't look at me" are no request).
_EN_AT_ME = re.compile(
    r"\b(?:look|glance|stare|gaze|turn|face|eyes?)\s+(?:right\s+|back\s+|"
    r"straight\s+)?(?:(?:at|to|toward|towards|on|into)\s+)?(?:to\s+face\s+)?"
    r"(?:me|us|the\s+camera|camera|the\s+lens|the\s+screen|the\s+chat|"
    r"chat|the\s+viewers?)\b|\blook\s+(?:this\s+way|over\s+here|here)\b",
    re.I)
_EN_NOT = re.compile(r"\b(?:don'?t|do\s+not|dont|never|stop|quit|not|no|"
                     r"can'?t|cannot|won'?t|without|avoid|why|are\s+you|"
                     r"were\s+you|you'?re|keep)\b[^.!?\n]{0,25}$", re.I)
_JA_AT_ME = re.compile(
    r"(?:こっち|こちら|そっち|私|わたし|あたし|僕|ぼく|俺|おれ|カメラ|ｶﾒﾗ|"
    r"チャット|画面|みんな)(?:の方|のほう|側)?(?:を|に|の方を|のほうを)?"
    r"(?:見て|みて|見ろ|見なさい|向いて|むいて)(?![いるたな])")
_ZH_AT_ME = re.compile(
    r"(?:看着|看著|看向|望着|望著|盯着|盯著|朝着|朝著|朝|面向|面对着|面对|"
    r"面對|转向|轉向|看)(?:着|著)?(?:我|镜头|鏡頭|摄像头|攝像頭|屏幕|螢幕|"
    r"这边|這邊|这里|這裡|聊天室|弹幕|彈幕)(?!的|怎)")
_ZH_NOT = re.compile(r"(?:别|別|不要|不|莫|勿|甭|为什么|為什麼|干嘛)[^。！？\n]{0,3}$")
_JA_NOT = re.compile(r"(?:見ない|見ちゃだめ|見るな)")


def asked_at_me(line) -> bool:
    """The line asks her to look at the user, the camera or the chat (en
    "look at me", "face the camera"; ja こっち見て, 私を見て, カメラ見て; zh
    看着我, 看我, 看镜头). False for a negation ("don't look at me", 別看我),
    a question about her looking, and a line that names a side (asked_side:
    "look to your left" is look_sign's)."""
    s = line or ""
    if asked_side(s) is not None:
        return False
    for m in _EN_AT_ME.finditer(s):
        if not _EN_NOT.search(s[max(0, m.start() - 30):m.start()]):
            return True
    for m in _ZH_AT_ME.finditer(s):
        if not _ZH_NOT.search(s[max(0, m.start() - 4):m.start()]):
            return True
    return bool(_JA_AT_ME.search(s)) and not _JA_NOT.search(s)


# ----------------------------------------------- a place near her (glance)
# The user puts or points at something around her without asking her to look
# ("I put the candy basket at your right foot", "behind you", "on your
# left"; ja 足元, 右足, 後ろ, 左側に置いた; zh 你脚边, 你右边, 身后): a person
# glances there first. mentioned_place() reads the line, glance_targets()
# gives the glance beat, looks_toward() asks whether a pose already does it.
_EN_PLACE_VERB = (r"(?:at|by|near|next\s+to|beside|under|below|on|around|"
                  r"put|placed?|set|dropped|left|laid|hid|hidden|there'?s|"
                  r"is|are)")
_EN_BODY = (r"hand|hands|arm|arms|elbow|shoulder|shoulders|eye|eyes|cheek|"
            r"ear|ears|leg|legs|knee|knees|wrist|finger|fingers|brow|brows|"
            r"hair|nose|mouth|lip|lips|face|head|neck|ribbon|dress|skirt")
_EN_FEET = re.compile(
    r"\b" + _EN_PLACE_VERB + r"\b[^.!?\n]{0,30}?\byour\s+(?:(right|left)\s+)?"
    r"(?:foot|feet|toes?|shoes?|boots?|socks?)\b|\bat\s+your\s+feet\b", re.I)
_EN_SIDE_AT = re.compile(
    r"\b(?:on|at|to|by|near|in)\s+your\s+(right|left)(?:\s+side)?\b(?!\s+(?:"
    + _EN_BODY + r"|foot|feet)\b)", re.I)
_EN_BEHIND = re.compile(r"\bbehind\s+(?:you|your\s+back)\b", re.I)
_EN_ABOVE = re.compile(r"\b(?:above|over)\s+(?:you|your\s+head)\b|"
                       r"\bon\s+(?:top\s+of\s+)?your\s+head\b", re.I)
_EN_BELOW = re.compile(r"\b(?:below|beneath|under)\s+you\b", re.I)
_JA_OWN = ("私の", "僕の", "俺の", "わたしの", "あたしの", "ぼくの", "自分の")
_JA_FEET = re.compile(r"足元|足下(?=[にへの])|(?:(右|左)足)(?:元)?(?:の(?:近く|そば|横|あたり|ところ))?(?=[にへ])")
_JA_BEHIND = re.compile(r"(?:あなた|君|きみ|キミ|お前)の?(?:後ろ|うしろ|背後)|"
                        r"(?<!私の)(?<!僕の)(?<!俺の)(?<!わたしの)"
                        r"(?<!髪の)(?:後ろ|うしろ|背後)(?=[にへで])")
_JA_SIDE = re.compile(r"(右|左)(?:側|がわ|の方|のほう)?に(?=置|あ|い|立|落|隠)")
_JA_ABOVE = re.compile(r"頭の上|頭上|あたまの上")
_ZH_FEET = re.compile(r"(?:你(?:的)?(右|左)?)?[脚腳](?:边|邊|下|旁)")
_ZH_BEHIND = re.compile(r"(?<!我)(?<!我的)(?:身[后後]|[背][后後]|你(?:的)?[后後][面边邊])")
_ZH_SIDE = re.compile(r"你(?:的)?(右|左)(?:手边|手邊|边|邊|侧|側|面)")
_ZH_ABOVE = re.compile(r"你(?:的)?(?:头顶|頭頂|头上|頭上|上方|上面)")
# A look word anywhere: asked_side also reads "your side" (你右边, 君の左側)
# without one, which is a place here.
_LOOK_VERB = re.compile(r"見[てろるえな]|向[いけくこ]|視線|[看望转轉瞧朝]")


def _owned(s, start, owners) -> bool:
    return any(s[:start].endswith(o) for o in owners)


def mentioned_place(line):
    """{"side": -1 (her right) | +1 (her left) | 0, "vert": -1 (down, her
    feet) | +1 (up) | 0, "behind": bool} for a line that puts or points at
    something near her without asking her to look (en "at your right foot",
    "on your left", "behind you", "above you"; ja 足元, 右足に, 後ろに, 左側に
    置いた; zh 你脚边, 你右边, 身后); None for any other line: an explicit
    look request (asked_side, asked_at_me, a look / turn / 見て / 看 word),
    the user's own side ("on my right", "my right hand hurts", "turn right
    at the station") or a part of her body ("your right hand"). Her right
    is negative (the look_sign convention). Unlike asked_side, "your
    side" without a look word (你右边, 君の左側) is a place."""
    s = (line or "").translate(str.maketrans({"’": "'", "‘": "'"}))
    if not s.strip() or asked_at_me(s) or _EN_LOOK.search(s) \
            or _LOOK_VERB.search(s):
        return None
    sides, vert, behind = set(), set(), False

    def side(w):
        if w:
            sides.add(-1 if w.lower() in ("right", "右") else 1)
    for m in _EN_FEET.finditer(s):
        vert.add(-1)
        side(m.group(1))
    for m in _EN_SIDE_AT.finditer(s):
        side(m.group(1))
    behind |= bool(_EN_BEHIND.search(s))
    if _EN_ABOVE.search(s):
        vert.add(1)
    if _EN_BELOW.search(s):
        vert.add(-1)
    for m in _JA_FEET.finditer(s):
        if not _owned(s, m.start(), _JA_OWN):
            vert.add(-1)
            side(m.group(1))
    for m in _JA_SIDE.finditer(s):
        if not _owned(s, m.start(), _JA_OWN):
            side(m.group(1))
    for m in _JA_BEHIND.finditer(s):
        if not _owned(s, m.start(), _JA_OWN):
            behind = True
    if _JA_ABOVE.search(s):
        vert.add(1)
    for m in _ZH_FEET.finditer(s):
        if not _owned(s, m.start(), ("我", "我的")):
            vert.add(-1)
            side(m.group(1))
    for m in _ZH_SIDE.finditer(s):
        side(m.group(1))
    behind |= bool(_ZH_BEHIND.search(s))
    if _ZH_ABOVE.search(s):
        vert.add(1)
    if not (sides or vert or behind):
        return None
    return {"side": next(iter(sides)) if len(sides) == 1 else 0,
            "vert": next(iter(vert)) if len(vert) == 1 else 0,
            "behind": behind}


GLANCE_DUR, GLANCE_HOLD = 0.3, 0.45          # seconds: ease in, then hold
PITCH_MIN, GAZE_Y_MIN = 6.0, 0.25
BEHIND_YAW, BEHIND_GAZE = 15.0, 0.5


def glance_targets(place) -> dict:
    """The pose targets of the glance at `place`: feet / below: head.pitch
    down and gaze.y down (with the side's yaw and gaze.x when it has one);
    a side: head.yaw and gaze.x to it (her right negative); above: up;
    behind: over her right shoulder (-25 degrees)."""
    t = {}
    if place.get("behind"):
        return {"head.yaw": -25, "gaze.x": -0.6}
    sd, vt = place.get("side", 0), place.get("vert", 0)
    if sd:
        t["head.yaw"] = sd * (8 if vt else 14)
        t["gaze.x"] = sd * (0.3 if vt else 0.5)
    if vt:
        t["head.pitch"] = -12 if vt < 0 else 10
        t["gaze.y"] = vt * 0.5
    return t


def looks_toward(tg, place) -> bool:
    """A pose's targets that already look toward `place`: any component the
    place has (the side's head.yaw or gaze.x, the up / down head.pitch or
    gaze.y, behind: a clear turn either way) is met."""
    yaw, gx = _num(tg.get("head.yaw")), _num(tg.get("gaze.x"))
    pit, gy = _num(tg.get("head.pitch")), _num(tg.get("gaze.y"))
    sd, vt = place.get("side", 0), place.get("vert", 0)
    if place.get("behind") and (
            (yaw is not None and abs(yaw) >= BEHIND_YAW)
            or (gx is not None and abs(gx) >= BEHIND_GAZE)):
        return True
    if sd and ((yaw is not None and yaw * sd >= YAW_MIN)
               or (gx is not None and gx * sd >= GAZE_MIN)):
        return True
    return bool(vt and ((pit is not None and pit * vt >= PITCH_MIN)
                        or (gy is not None and gy * vt >= GAZE_Y_MIN)))


AT_ME_YAW = 3.0          # centred on the viewer: |head.yaw| degrees (DPO)
AT_ME_GAZE = 0.15        # and |gaze.x|


def final_look(rec):
    """(head.yaw, gaze.x) the reply ends on: the last pose marker that
    sets each; None for one it never sets."""
    yaw = gx = None
    for t in _poses(rec):
        tg = t["arg"]["targets"]
        if _num(tg.get("head.yaw")) is not None:
            yaw = _num(tg["head.yaw"])
        if _num(tg.get("gaze.x")) is not None:
            gx = _num(tg["gaze.x"])
    return yaw, gx


def asked_wave(line) -> bool:
    """The line asks to wave (flow.handask.asked_wave_action)."""
    return ha.asked_wave_action(line)


def asked_hands(line, topic=None, quiet=False) -> bool:
    """Whether the line explicitly asks the character for a hand or arm
    action (flow.handask.asked_hand_action): by its words, not by its topic
    (an invitation or a tease is no request; show / pose also need a limb
    word), not for the head or the eyes alone, not in the past or as a
    compliment."""
    return not quiet and ha.asked_hand_action(line, topic)


# -------------------------------------------------------------- the reply

def _poses(rec):
    return [t for t in rec.markers if t["tag"] == pz.POSE]


def _hand_keys(t) -> bool:
    return any(dx.is_hand(k) for k in t["arg"]["targets"])


def has_hand(rec) -> bool:
    """A hand target in a pose, a move or a gesture."""
    return any((t["tag"] == pz.POSE and _hand_keys(t))
               or t["tag"] in ("move", "gesture") for t in rec.markers)


def has_wave(rec) -> bool:
    return any(t["tag"] == "move" and (t["arg"] or {}).get("name") == "wave"
               for t in rec.markers) or any(
        t["tag"] == pz.POSE and _hand_keys(t) for t in rec.markers)


def first_turn(rec):
    """The sign of the first pose marker with a head.yaw of YAW_MIN or a
    gaze.x of GAZE_MIN: -1 / +1, or 0 when the marker's two disagree; None
    when no marker turns."""
    for t in _poses(rec):
        tg = t["arg"]["targets"]
        signs = []
        yaw, gx = _num(tg.get("head.yaw")), _num(tg.get("gaze.x"))
        if yaw is not None and abs(yaw) >= YAW_MIN:
            signs.append(1 if yaw > 0 else -1)
        if gx is not None and abs(gx) >= GAZE_MIN:
            signs.append(1 if gx > 0 else -1)
        if signs:
            return signs[0] if len(set(signs)) == 1 else 0
    return None


def _round(v):
    n = _num(v)
    if n is None:
        return str(v)
    return round(n) if abs(n) >= 2 else round(n, 1)


def opening(rec):
    """(variable set, rounded values) of the first pose marker; None
    without a pose. dur and hold are not part of it."""
    for t in _poses(rec):
        tg = t["arg"]["targets"]
        keys = tuple(sorted(tg))
        return keys, tuple((k, _round(tg[k])) for k in keys)
    return None


def is_q8r2_opening(rec) -> bool:
    for t in _poses(rec):
        tg = t["arg"]["targets"]
        if frozenset(tg) != Q8R2_KEYS:
            return False
        for k, (lo, hi) in Q8R2_RANGE.items():
            v = _num(tg[k])
            if v is None or not lo - 1e-6 <= v <= hi + 1e-6:
                return False
        return True
    return False


def signature(rec) -> tuple:
    """The sequence of markers by their variable sets (the tag alone for
    the others): what a collapsed reply repeats."""
    sig = []
    for t in rec.markers:
        if t["tag"] == pz.POSE:
            sig.append(tuple(sorted(t["arg"]["targets"])))
        else:
            sig.append(t["tag"])
    return tuple(sig)


def mixed(text) -> bool:
    """A line that mixes scripts: two of Latin, kana / han and hangul
    letters, the smaller with at least 4 letters and a quarter of all."""
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    cjk = sum(1 for c in text if 0x3040 <= ord(c) <= 0x30FF
              or 0x3400 <= ord(c) <= 0x9FFF)
    hangul = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)
    tot = latin + cjk + hangul
    second = sorted((latin, cjk, hangul))[1]
    return tot > 0 and second >= 4 and second / tot >= 0.25


QUIET_SAYS = [r"chat (?:is|'s|has gone|was) (?:so |very |really |a bit )?"
              r"(?:quiet|silent|dead)", r"no ?one (?:is|'s) (?:talking|"
              r"saying|writing|chatting)", r"nobody (?:is|'s) (?:talking|"
              r"saying|writing|chatting)", r"quiet in (?:the )?chat",
              r"チャット.{0,12}(?:静か|しずか|シーン)",
              r"没人说话|沒人說話|没有人说话|沒有人說話",
              r"(?:聊天室|聊天|弹幕|彈幕|评论区|留言).{0,8}(?:安静|安靜|冷清)"]


def says_quiet(spoken) -> bool:
    return bool(_words(QUIET_SAYS, spoken or ""))


# ---------------------------------------------------------------- checks
# Each takes (records, bodies) and returns ({id: verdict}, extra stats,
# {id: reason}); the distributions give their numbers as extras.

def check_look_sign(recs, bodies):
    v, why = {}, {}
    for r in recs:
        side = asked_side(r["user"])
        if side is None:
            v[r["id"]] = NA
            continue
        want = -1 if side == "R" else 1
        got = first_turn(r)
        if got == want:
            v[r["id"]] = PASS
        else:
            v[r["id"]] = FAIL
            why[r["id"]] = "no_turn" if got is None else "wrong_sign"
    return v, dict(Counter(why.values())), why


def check_look_at_me(recs, bodies):
    v, why = {}, {}
    for r in recs:
        if r.get("quiet") or not asked_at_me(r["user"]):
            v[r["id"]] = NA
            continue
        yaw, gx = final_look(r)
        bad_yaw = yaw is not None and abs(yaw) >= YAW_MIN
        bad_gaze = gx is not None and abs(gx) >= GAZE_MIN
        if bad_yaw or bad_gaze:
            v[r["id"]] = FAIL
            why[r["id"]] = ("both" if bad_yaw and bad_gaze
                            else "yaw" if bad_yaw else "gaze")
        else:
            v[r["id"]] = PASS
    return v, dict(Counter(why.values())), why


def check_glance_at_mention(recs, bodies):
    """The user puts something near her without asking her to look
    (mentioned_place): the first or second pose marker must move her head
    or gaze toward it."""
    v, why = {}, {}
    for r in recs:
        place = None if r.get("quiet") else mentioned_place(r["user"])
        if place is None:
            v[r["id"]] = NA
            continue
        poses = _poses(r)[:2]
        if any(looks_toward(t["arg"]["targets"], place) for t in poses):
            v[r["id"]] = PASS
        else:
            v[r["id"]] = FAIL
            why[r["id"]] = "no_pose" if not poses else "no_glance"
    return v, dict(Counter(why.values())), why


# ---------------------------------------------------------- corrections
# The user fixes a word she said ("not X but Y"): X must not come back.

_Q = "[\"'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f]?"
_W = r"([\w'\u2019-]+|\"[^\"]{1,30}\"|\u201c[^\u201d]{1,30}\u201d)"
_ENX = [  # (regex, index of X, index of Y)
    (r"\bnot\s+" + _Q + _W + _Q + r"\s*[,;-]?\s*(?:but|rather|it'?s|its|"
     r"it\s+is|that'?s)\s+" + _Q + _W, 1, 2),
    (r"\b(?:it'?s|its|it\s+is|that'?s|is|say|called|spell(?:ed)?)\s+" + _Q
     + _W + _Q + r"\s*[,;-]?\s*(?:and\s+)?not\s+" + _Q + _W, 2, 1),
    (r"\byou\s+(?:said|called|wrote|spelled|spelt)\s+" + _Q + _W + _Q
     + r"[^.!?\n]{0,15}?\b(?:it'?s|its|it\s+is|is|but)\s+" + _Q + _W, 1, 2),
    (r"^\s*" + _Q + _W + _Q + r"\s*[,;!.-]*\s*(?:but\s+)?you\s+(?:said|wrote|"
     r"called\s+it|spelled\s+it|spelt\s+it)\s+" + _Q + _W, 2, 1),
]
JA_ZH_MAX_CHARS = 40  # the ja / zh clause forms read only short lines
_T = r"([^\s\u3001\u3002\uff01!\uff1f?,\uff0c\u300c\u300d\u300e\u300f]{1,20}?)"
_NEG = r"(?:じゃない|ではない|じゃなくて|ではなくて|じゃなく|ではなく|じゃありません|ではありません)"
_COP = r"(?:だよ|です|だ|でしょ|だってば|だぞ|だよね|だね)"
_SEP = r"[\s\u3001\u3002\uff01!\uff0c,]*"
_JAX = [
    (_T + _COP + _SEP + _T + _NEG, 2, 1),               # Yだ、Xじゃない
    (_T + _NEG + _SEP + _T + r"(?:" + _COP + r"|$|[!\uff01\u3002])", 1, 2),
]
_ZHX = [
    (r"不是\s*" + _Q + r"([^\s，。！？,!?「」]{1,12}?)" + _Q
     + r"[，,\s]*(?:而是|应该是|應該是|该是|是|叫)\s*" + _Q
     + r"([^\s，。！？,!?「」]{1,12})", 1, 2),
    (r"(?:应该是|應該是|是|叫)\s*" + _Q + r"([^\s，。！？,!?「」不]{1,12}?)"
     + _Q + r"[，,\s]*(?:而)?不是\s*" + _Q + r"([^\s，。！？,!?「」]{1,12})", 2, 1),
]
_ZH_ONLY_X = re.compile(r"你(?:说|說|讲|講|写|寫)(?:的|成)?\s*" + _Q
                        + r"([^\s，。！？,!?「」]{1,12}?)" + _Q
                        + r"[，,\s]*(?:不对|不對|错了|錯了|不是)")
_CUE = re.compile(r"違う|ちがう|間違|まちがい|不对|不對|"
                  r"错了|錯了|\bwrong\b|^\s*no\b[,!.]|that'?s not (?:it|right|how)|"
                  r"\bspell|\bpronounce|\bsaid it wrong", re.I)
CUE_MAX_CHARS = 60    # the fuzzy fallback reads only short correcting lines
CUE_RATIO = 0.7       # difflib ratio from here (below: unrelated words)
CUE_MIN_LEN = 4       # words shorter than this are never a corrected word
_TOKS = re.compile(r"[A-Za-z][A-Za-z'-]{2,}|[\u30a0-\u30ff\u3040-\u309f"
                   r"\u4e00-\u9fffー]{2,}")


def _clean_word(w):
    w = (w or "").strip()
    if len(w) >= 2 and w[0] in "\"\u201c" and w[-1] in "\"\u201d":
        w = w[1:-1].strip()
    return w.strip("\"'\u201c\u201d\u2018\u2019\u300c\u300d")


def corrections(line, prev=None) -> list:
    """[(X, Y)]: the user's correction of a word, X the wrong form (always
    set) and Y the right one (None when the line gives only X). Explicit
    forms first (en "not X but Y", "it's Y, not X", "you said X, it's Y";
    ja "XじゃなくてY", "Yだ、Xじゃない"; zh "不是X，是Y", "是Y不是X", "你说X不对");
    else, when `prev` (her last reply's spoken text) is given and the line
    has a correcting cue (違う, wrong, 不对 ...), each word of the line that
    is close (difflib ratio 0.6-0.95) to a different word of `prev` is Y
    for that word as X. The Japanese forms capture the clause before the
    particle (no word boundaries there): see wrong_forms."""
    s = line or ""
    out = []

    def add(x, y):
        x, y = _clean_word(x), _clean_word(y) if y else None
        if x and len(x) >= 2 and (not y or len(y) >= 2 or not y.isascii()) and x != y and (x, y) not in out \
                and not (y and (x in y or y in x)):
            out.append((x, y))
    for pat, ix, iy in _ENX:
        for m in re.finditer(pat, s, re.I):
            add(m.group(ix), m.group(iy))
    for pat, ix, iy in (_JAX + _ZHX if len(s) <= JA_ZH_MAX_CHARS else []):
        for m in re.finditer(pat, s):
            add(m.group(ix), m.group(iy))
    for m in _ZH_ONLY_X.finditer(s):
        add(m.group(1), None)
    if out or not prev or len(s) > CUE_MAX_CHARS or not _CUE.search(s):
        return out
    import difflib
    ptoks = {x for x in _TOKS.findall(prev) if len(x) >= CUE_MIN_LEN}
    for y in _TOKS.findall(s):
        if len(y) < CUE_MIN_LEN or y in prev:
            continue
        best = None
        for x in ptoks:
            if x == y or x in s:
                continue
            r = difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio()
            if CUE_RATIO <= r <= 0.95 and (best is None or r > best[0]):
                best = (r, x)
        if best:
            add(best[1], y)
    return out


def wrong_forms(x, y=None) -> list:
    """The strings that count as the reply repeating the wrong X: X itself
    and, for a Japanese / Chinese clause that may carry words before the
    wrong word, its tails of at least min(len, 4) characters; never a string
    the right form Y contains."""
    forms = [x]
    if not x.isascii() and len(x) > 4:
        forms += [x[i:] for i in range(1, len(x) - 3)]
    return [f for f in forms if not (y and f.lower() in y.lower())]


def says_wrong(spoken, x, y=None) -> bool:
    t = (spoken or "").lower()
    return any(f.lower() in t for f in wrong_forms(x, y))


def check_correction_obeyed(recs, bodies):
    v, by_s = {}, defaultdict(list)
    for r in recs:
        by_s[r["session"]].append(r)
    for rs in by_s.values():
        prev = None
        for r in sorted(rs, key=lambda r: r["turn"]):
            pairs = [] if r["quiet"] else corrections(
                r["user"], prev["spoken"] if prev else None)
            if not pairs:
                v[r["id"]] = NA
            else:
                v[r["id"]] = FAIL if any(says_wrong(r["spoken"], x, y)
                                         for x, y in pairs) else PASS
            prev = r
    return v, {}, {}


# ------------------------------------------------------------- pose copy

COPY_PREV = 3         # replies back that count
COPY_SHARE = 0.75     # of the beats equal in place, at least


def pose_seq(rec) -> tuple:
    """The pose markers of a reply as ((variable, rounded value) ...) each,
    dur and hold aside, in order."""
    out = []
    for t in _poses(rec):
        tg = t["arg"]["targets"]
        out.append(tuple((k, _round(tg[k])) for k in sorted(tg)))
    return tuple(out)


def copies(cur, prev) -> bool:
    """cur's pose sequence repeats prev's: with n the longer length (at
    least 2), at least max(2, ceil(0.75 n)) of the poses equal in place."""
    n = max(len(cur), len(prev))
    if min(len(cur), len(prev)) < 2:
        return False
    same = sum(1 for a, b in zip(cur, prev) if a == b)
    return same >= max(2, -(-int(COPY_SHARE * 100) * n // 100))


def check_pose_copy(recs, bodies):
    v, by_s = {}, defaultdict(list)
    for r in recs:
        by_s[r["session"]].append(r)
    for rs in by_s.values():
        seen = []
        for r in sorted(rs, key=lambda r: r["turn"]):
            cur = pose_seq(r)
            if len(cur) < 2 or not seen:
                v[r["id"]] = NA
            else:
                v[r["id"]] = FAIL if any(copies(cur, p)
                                         for p in seen[-COPY_PREV:]) else PASS
            seen.append(cur)
    return v, {}, {}


def check_wave_obeyed(recs, bodies):
    v = {}
    for r in recs:
        if not asked_wave(r["user"]) or not bodies.has_hands(r["avatar"]):
            v[r["id"]] = NA
        else:
            v[r["id"]] = PASS if has_wave(r) else FAIL
    return v, {}, {}


def check_body_request_obeyed(recs, bodies):
    v = {}
    for r in recs:
        if (r["quiet"] or not bodies.has_hands(r["avatar"])
                or not asked_hands(r["user"], r.get("topic"))):
            v[r["id"]] = NA
        else:
            v[r["id"]] = PASS if has_hand(r) else FAIL
    return v, {}, {}


def check_opening(recs, bodies):
    v, sets, exact = {}, Counter(), Counter()
    for r in recs:
        o = opening(r)
        if o is None:
            v[r["id"]] = NA
            continue
        sets[o[0]] += 1
        exact[o] += 1
        v[r["id"]] = FAIL if is_q8r2_opening(r) else PASS
    n = sum(sets.values())
    extra = {}
    if n:
        k, c = sets.most_common(1)[0]
        extra["top_set"] = {"share": c / n, "set": list(k)}
        k, c = exact.most_common(1)[0]
        extra["top_exact"] = {"share": c / n, "opening": dict(k[1])}
    return v, extra, {}


def check_template_repeat(recs, bodies):
    v = {}
    by_s = defaultdict(list)
    for r in recs:
        by_s[r["session"]].append(r)
    for rs in by_s.values():
        prev = None
        for r in sorted(rs, key=lambda r: r["turn"]):
            sig = signature(r)
            if not sig or prev is None:
                v[r["id"]] = NA
            else:
                v[r["id"]] = FAIL if sig == prev else PASS
            prev = sig or prev
    return v, {}, {}


def check_language(recs, bodies):
    v = {}
    for r in recs:
        u = r["user"]
        lu, lr = pz.detect_lang(u), pz.detect_lang(r["spoken"])
        if (r["quiet"] or r.get("mix") or mixed(u) or lu is None
                or lr is None):
            v[r["id"]] = NA
        else:
            v[r["id"]] = PASS if lu == lr else FAIL
    return v, {}, {}


def _first_index(markers, pred):
    for i, t in enumerate(markers):
        if pred(t):
            return i
    return None


def _is_full(t):
    return t["tag"] == pz.FRAME and t["arg"] == "full"


def check_frame_for_dance(recs, bodies):
    v = {}
    for r in recs:
        i = _first_index(r.markers, lambda t: t["tag"] == "move"
                         and (t["arg"] or {}).get("name") in fr.WHOLE_BODY)
        if (i is None or r["start"] == "full"
                or not bodies.can_frame(r["avatar"])):
            v[r["id"]] = NA
        else:
            v[r["id"]] = PASS if any(_is_full(t) for t in r.markers[:i]) \
                else FAIL
    return v, {}, {}


def check_frame_before_wave(recs, bodies):
    v = {}
    for r in recs:
        i = _first_index(r.markers, lambda t: t["tag"] == "move"
                         and (t["arg"] or {}).get("name") == "wave")
        if i is None:
            v[r["id"]] = NA
        else:
            v[r["id"]] = FAIL if any(_is_full(t) for t in r.markers[:i]) \
                else PASS
    return v, {}, {}


def check_beats(recs, bodies):
    v, sent, beats, n = {}, 0, 0, 0
    for r in recs:
        c = dx.beat_counts(r["raw"])
        if not c["sentences"]:
            v[r["id"]] = NA
            continue
        n += 1
        sent += c["sentences"]
        beats += c["beats"]
        v[r["id"]] = PASS if c["beats"] >= c["sentences"] else FAIL
    return v, {"per_sentence": beats / sent if sent else None,
               "per_reply": beats / n if n else None}, {}


def check_quiet_says_quiet(recs, bodies):
    v = {}
    for r in recs:
        if not r["quiet"]:
            v[r["id"]] = NA
        else:
            v[r["id"]] = FAIL if says_quiet(r["spoken"]) else PASS
    return v, {}, {}


CHECKS = {"look_sign": check_look_sign, "look_at_me": check_look_at_me,
          "glance_at_mention": check_glance_at_mention,
          "correction_obeyed": check_correction_obeyed,
          "pose_copy": check_pose_copy, "wave_obeyed": check_wave_obeyed,
          "body_request_obeyed": check_body_request_obeyed,
          "opening": check_opening, "template_repeat": check_template_repeat,
          "language": check_language, "frame_for_dance": check_frame_for_dance,
          "frame_before_wave": check_frame_before_wave, "beats": check_beats,
          "quiet_says_quiet": check_quiet_says_quiet}


# ------------------------------------------------------------- aggregate

def aggregate(recs, bodies, names=None) -> dict:
    """{check: {pass, fail, na, n, fail_rate, extras..., fail_ids}} for
    these records."""
    out = {}
    for name in names or CHECKS:
        v, extra, _why = CHECKS[name](recs, bodies)
        c = Counter(v.values())
        n = c[PASS] + c[FAIL]
        out[name] = {"pass": c[PASS], "fail": c[FAIL], "na": c[NA], "n": n,
                     "fail_rate": c[FAIL] / n if n else None, **extra,
                     "fail_ids": [i for i, x in v.items()
                                  if x == FAIL][:FIDS]}
    return out


def run(recs, bodies, by=None, names=None) -> dict:
    """{"n", "checks", "style", "by_avatar": {avatar: {"n", "checks",
    "style"}}}. "style" is flow.style's numbers (not pass / fail checks)."""
    res = {"n": len(recs), "checks": aggregate(recs, bodies, names),
           "style": sty.style(recs)}
    if by == "avatar":
        groups = defaultdict(list)
        for r in recs:
            groups[r["avatar"]].append(r)
        res["by_avatar"] = {a: {"n": len(g),
                                "checks": aggregate(g, bodies, names),
                                "style": sty.style(g)}
                            for a, g in sorted(groups.items())}
    return res


# ----------------------------------------------------------------- table

def _rate(c):
    return None if not c or c["fail_rate"] is None else (
        c["fail_rate"], c["fail"], c["n"])


def rows_of(ck) -> list:
    """[(label, value)] for one input's checks: a fail rate is (rate, fail,
    n), a share or a mean a float."""
    op = ck.get("opening") or {}
    ls = ck.get("look_sign")
    return [
        ("look_sign", _rate(ls)),
        ("  wrong sign / no turn",
         (ls.get("wrong_sign", 0), ls.get("no_turn", 0)) if ls else None),
        ("look_at_me", _rate(ck.get("look_at_me"))),
        ("glance_at_mention", _rate(ck.get("glance_at_mention"))),
        ("correction_obeyed", _rate(ck.get("correction_obeyed"))),
        ("pose_copy", _rate(ck.get("pose_copy"))),
        ("wave_obeyed", _rate(ck.get("wave_obeyed"))),
        ("body_request_obeyed", _rate(ck.get("body_request_obeyed"))),
        ("opening = q8r2 template", _rate(op)),
        ("opening top set share", (op.get("top_set") or {}).get("share")),
        ("opening top exact share",
         (op.get("top_exact") or {}).get("share")),
        ("template_repeat", _rate(ck.get("template_repeat"))),
        ("language", _rate(ck.get("language"))),
        ("frame_for_dance", _rate(ck.get("frame_for_dance"))),
        ("frame_before_wave", _rate(ck.get("frame_before_wave"))),
        ("beats < sentences", _rate(ck.get("beats"))),
        ("beats per sentence", (ck.get("beats") or {}).get("per_sentence")),
        ("beats per reply", (ck.get("beats") or {}).get("per_reply")),
        ("quiet_says_quiet", _rate(ck.get("quiet_says_quiet"))),
    ]


def _cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, tuple) and len(v) == 3:
        return f"{v[0]:.0%} ({v[1]}/{v[2]})"
    if isinstance(v, tuple):
        return " / ".join(str(x) for x in v)
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def table(labels, results, title=None) -> str:
    """The checks of several inputs side by side, one column each (fail
    rates, lower is better, as percent (fail/applicable); shares and means
    as numbers). `results` is run()'s dict or its by_avatar entry per
    input."""
    rows = [rows_of(r["checks"]) for r in results]
    if any((r.get("style") or {}).get("n") for r in results):
        # The style numbers under the checks (flow.style): one row per
        # number, "-" for an input without any.
        srows = [dict(sty.rows_of(r.get("style"))) for r in results]
        labs = next(sty.rows_of(r.get("style")) for r in results
                    if (r.get("style") or {}).get("n"))
        for lab, _v in labs:
            for row, sr in zip(rows, srows):
                row.append((lab, sr.get(lab)))
    w0 = max(len(lab) for lab, _ in rows[0]) + 2
    wc = max([18] + [len(x) + 2 for x in labels])
    out = [title] if title else []
    out.append(f"{'':{w0}s}" + "".join(f"{x:>{wc}s}" for x in labels))
    out.append(f"{'replies':{w0}s}" + "".join(
        f"{r['n']:>{wc}d}" for r in results))
    for k in range(len(rows[0])):
        out.append(f"{rows[0][k][0]:{w0}s}" + "".join(
            f"{_cell(r[k][1]):>{wc}s}" for r in rows))
    return "\n".join(out)


def tables(labels, results, by=None) -> str:
    """The whole report: the overall table, and with by == "avatar" one per
    avatar (an input without it shows "-")."""
    text = table(labels, results)
    if by == "avatar":
        names = sorted({a for r in results for a in r.get("by_avatar", {})})
        for a in names:
            empty = {"n": 0, "checks": {}}
            sub = [r.get("by_avatar", {}).get(a, empty) for r in results]
            text += "\n\n" + table(labels, sub, title=f"[{a}]")
    return text


# ------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", metavar="PATH",
                    help="an evalsft student.jsonl (or its folder), a "
                         "flow.dataset folder or shard, a flow.live "
                         "turns.jsonl")
    ap.add_argument("--by", default=None, choices=("avatar",),
                    help="also one table per avatar")
    ap.add_argument("--json", default=None, help="write the numbers (and "
                    f"the first {FIDS} failing ids of each check) here")
    ap.add_argument("--data", default=None, help="a flow.dataset folder: "
                    "the avatars' motion budgets from its avatars.json "
                    "[else rigs/NAME.body.json]")
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--avatar", default=None, help="the avatar of a live "
                    "turns.jsonl (it has none)")
    ap.add_argument("--checks", default=None, help="comma list of "
                    f"{', '.join(CHECKS)} [all]")
    a = ap.parse_args(argv)
    names = [x for x in (a.checks or "").split(",") if x] or None
    for x in names or ():
        if x not in CHECKS:
            raise SystemExit(f"--checks {x!r}: one of {', '.join(CHECKS)}")
    bodies = Bodies(a.data, a.rigs)
    labels, results = [], []
    for p in a.paths:
        for label, recs in load(p, a.avatar):
            if label in labels:
                # The same label twice (two runs' student.jsonl): the path.
                label = os.path.normpath(p) + (
                    ":" + label.rsplit(":", 1)[-1] if ":" in label else "")
                while label in labels:
                    label += "'"
            labels.append(label)
            results.append(run(recs, bodies, a.by, names))
    print(tables(labels, results, a.by))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(zip(labels, results)), fh, ensure_ascii=False,
                      indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
