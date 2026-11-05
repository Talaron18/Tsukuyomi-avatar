#!/usr/bin/env python3
"""
A small rule graph with reminders for the live student (flow.live
--reminders auto|off, default off) and for the prompt A/B of
flow.train.evalsft (docs/embodiment.md).

The beats2 prompt is long (~3-4k tokens) and the student still breaks rules
it states: left and right flipped, a sideways head on "look at me", no glance
at a place the viewer mentions. A rule that matters once in twenty turns sits
in the middle of the prompt, the worst place to be read from (docs/research.md
section 4). Here each rule is a node of flow/rules.json:

    {"id", "trigger" (a detector of DETECTORS), "text" (one short line, the
     beats2 prompt's own wording, flow/pose.py), "needs" ([ids])}

Per turn, select():

  1. the nodes whose detector fires on the turn (the user line, her previous
     reply, the memory notes), then for each, in file order, the node with
     its `needs` (recursively);
  2. at most CAP (3) nodes in all: a triggered node comes together with its
     needs or not at all, so a node whose group no longer fits is skipped;
  3. deduplicated, and in the order of the file, which is the priority
     (look-side first, language last).

block() renders them as one line, "(reminders: <text> <text> <text>)", and
that is put with flow.memory.inject() where the memory notes go: the last
line(s) of the latest user message only, at send time. The history keeps the
plain message, the prefix stays byte-stable, the context budget counts the
block (flow.live puts it in the same note_block as the notes). Nothing is
added when no node fires.

A node whose trigger is "never" is only pulled in by another's needs. The
language node fires on a switch only (the user's language differs from her
previous reply's), or it would fill a slot every turn. The game and
correction nodes need the notes, so they stay silent without --memory (and in
the eval, which has no notes).
"""

from __future__ import annotations

import json
import os
import re

CAP = 3
PREFIX, SUFFIX = "(reminders: ", ")"
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.json")
MODES = ("auto", "off")
TEXT_MAX = 260                 # characters of one node's line

try:
    from . import checks as _checks
    from . import pose as pz
except ImportError:                       # run as a script
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from flow import checks as _checks
    from flow import pose as pz


# ----------------------------------------------------------- the detectors

class Ctx:
    """What the detectors read: the turn's user line (a quiet turn's is
    pose.QUIET_CHAT), her previous reply (raw, markers and all), the memory
    notes (flow.memory.Notes or None)."""

    def __init__(self, user, prev="", notes=None):
        self.user = user or ""
        self.prev = prev or ""
        self.notes = notes

    @property
    def quiet(self) -> bool:
        return self.user.strip() == pz.QUIET_CHAT

    @property
    def prev_spoken(self) -> str:
        try:
            from . import memory as mem
        except ImportError:
            from flow import memory as mem
        return mem.strip_markers(self.prev)


_DANCE = re.compile(
    r"\bdanc(?:e|es|ed|ing)\b|\bboogie\b|踊|ダンス|おどって|跳舞|舞蹈|"
    r"来段舞|跳个舞|跳支舞", re.I)
_DANCE_NOT = re.compile(
    r"\b(?:don'?t|do\s+not|not|never|stop|no)\b[^.!?\n]{0,12}\bdanc|"
    r"踊らない|踊るな|踊らなく|别跳|不要跳|不跳|別跳", re.I)
_DANCE_PAST = re.compile(
    r"\b(?:i|we|they|she|he)\s+(?:danced|was dancing|were dancing)\b|"
    r"\bdanc(?:e|ing)\s+(?:class|lesson|floor|party|club|practice|game)\b",
    re.I)


def asked_dance(ctx) -> bool:
    s = ctx.user
    return bool(not ctx.quiet and _DANCE.search(s)
                and not _DANCE_NOT.search(s) and not _DANCE_PAST.search(s))


def asked_side(ctx) -> bool:
    return _checks.asked_side(ctx.user) is not None


def asked_at_me(ctx) -> bool:
    return not ctx.quiet and _checks.asked_at_me(ctx.user)


def asked_wave(ctx) -> bool:
    return not ctx.quiet and _checks.asked_wave(ctx.user)


def mentioned_place(ctx) -> bool:
    return not ctx.quiet and _checks.mentioned_place(ctx.user) is not None


_NEVER = re.compile(r"\bnever\s+(.+?)\s*$", re.I)


def corrected(ctx) -> bool:
    """The line corrects a word of hers (flow.checks.corrections), or it
    brings back a wrong form the notes list ("right, never wrong")."""
    if ctx.quiet:
        return False
    if _checks.corrections(ctx.user, ctx.prev_spoken):
        return True
    for e in getattr(ctx.notes, "corrections", None) or ():
        m = _NEVER.search(e)
        if m and m.group(1).strip(" ,.;").lower() in ctx.user.lower():
            return True
    return False


def game_on(ctx) -> bool:
    return bool(getattr(ctx.notes, "game", None))


def chat_quiet(ctx) -> bool:
    return ctx.quiet


def language_switch(ctx) -> bool:
    """The user's language is not the one of her previous reply (both
    settled by pose.detect_lang; a mixed line or a quiet turn does not
    count)."""
    if ctx.quiet or not ctx.prev:
        return False
    lu = pz.detect_lang(ctx.user)
    lr = pz.detect_lang(ctx.prev_spoken)
    return bool(lu and lr and lu != lr and not _checks.mixed(ctx.user))


def never(ctx) -> bool:
    return False


DETECTORS = {f.__name__: f for f in (
    asked_side, asked_at_me, asked_wave, asked_dance, mentioned_place,
    corrected, game_on, chat_quiet, language_switch, never)}


# ------------------------------------------------------------- the graph

class Graph:
    """The nodes in priority order (file order)."""

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.byid = {}
        for n in self.nodes:
            for k in ("id", "trigger", "text", "needs"):
                if k not in n:
                    raise ValueError(f"rule node {n!r}: no {k!r}")
            if n["id"] in self.byid:
                raise ValueError(f"rule node {n['id']!r} twice")
            if n["trigger"] not in DETECTORS:
                raise ValueError(f"rule node {n['id']!r}: no detector "
                                 f"{n['trigger']!r}")
            if not n["text"].strip() or "\n" in n["text"] \
                    or len(n["text"]) > TEXT_MAX:
                raise ValueError(f"rule node {n['id']!r}: the text is one "
                                 f"short line (at most {TEXT_MAX} characters)")
            req = n.get("requires", [])
            if not isinstance(req, list) or not all(
                    isinstance(x, str) and ":" in x for x in req):
                raise ValueError(f"rule node {n['id']!r}: requires is a list "
                                 f"of capabilities like 'move:wave'")
            self.byid[n["id"]] = n
        for n in self.nodes:
            for m in n["needs"]:
                if m not in self.byid:
                    raise ValueError(f"rule node {n['id']!r} needs "
                                     f"{m!r}: no such node")
        self.rank = {n["id"]: i for i, n in enumerate(self.nodes)}

    def closure(self, nid) -> list:
        """nid and everything it needs, recursively, once each."""
        out, todo = [], [nid]
        while todo:
            x = todo.pop(0)
            if x not in out:
                out.append(x)
                todo += self.byid[x]["needs"]
        return out

    def fired(self, ctx) -> list:
        """The ids of the nodes whose detector fires, in priority order."""
        return [n["id"] for n in self.nodes
                if DETECTORS[n["trigger"]](ctx)]

    def allowed(self, nid, can) -> bool:
        """The node and all it needs ask nothing this body lacks (`can`:
        capabilities() of the body, or None: not known, all allowed)."""
        if can is None:
            return True
        return all(set(self.byid[x].get("requires", [])) <= can
                   for x in self.closure(nid))

    def select(self, ctx, cap=CAP, can=None) -> list:
        """The ids shown this turn (class docstring of the module). A node
        whose "requires" the body lacks is never shown: telling lafei to
        [move wave] only had the marker dropped (q8r3 A/B)."""
        chosen = []
        for nid in self.fired(ctx):
            if not self.allowed(nid, can):
                continue
            new = [x for x in self.closure(nid) if x not in chosen]
            if len(chosen) + len(new) <= cap:
                chosen += new
        return sorted(chosen, key=self.rank.get)

    def render(self, ids) -> str:
        if not ids:
            return ""
        return PREFIX + " ".join(self.byid[i]["text"] for i in ids) + SUFFIX


def load(path=None) -> Graph:
    with open(path or PATH, encoding="utf-8") as fh:
        return Graph(json.load(fh)["nodes"])


_GRAPH = None


def graph() -> Graph:
    """The shipped graph (flow/rules.json), read once."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = load()
    return _GRAPH


def capabilities(bm) -> set:
    """What a body offers that a rule node may require: "move:<name>" for
    each [move] it can do (flow.moves.offered), "frame:switch" when its
    camera can switch between bust and full (flow.framing.can_frame)."""
    try:
        from . import framing as fr, moves as mv
    except ImportError:
        from flow import framing as fr, moves as mv
    can = {f"move:{m}" for m in mv.offered(bm)}
    if fr.can_frame(bm):
        can.add("frame:switch")
    return can


def block(user, prev="", notes=None, g=None, cap=CAP, can=None):
    """(the "(reminders: ...)" block or "", the ids) for a turn. `prev`:
    her previous reply (raw), `notes`: flow.memory.Notes or None, `can`:
    capabilities() of the body (None: every node allowed)."""
    g = g or graph()
    ids = g.select(Ctx(user, prev, notes), cap, can)
    return g.render(ids), ids


def tail_block(text, prev_reply, notes_block="", notes=None, can=None,
               reminders="auto", g=None):
    """(the tail block of a turn, the rule ids): the memory notes block,
    then the "(reminders: ...)" line when `reminders` is "auto". The one
    function flow.live (Performer.tail_block) and the data generator
    (flow.dataset --reminders / --notes) both call, so what the student is
    trained on is what it is served. `notes`: flow.memory.Notes of the turn
    (the game and correction nodes read them), `can`: capabilities() of the
    body."""
    if reminders != "auto":
        return notes_block or "", []
    rem, ids = block(text, prev_reply, notes, g=g, can=can)
    return join(notes_block, rem), ids


def join(*blocks) -> str:
    """The blocks that are not empty, one per line: the tail block flow.live
    sends (the memory notes first, then the reminders)."""
    return "\n".join(b for b in blocks if b)
