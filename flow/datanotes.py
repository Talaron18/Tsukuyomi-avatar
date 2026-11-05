#!/usr/bin/env python3
"""
Memory notes for the teacher's data (flow.dataset --notes auto): the
"(stream notes: ...)" line the live student reads (flow.memory), built here
deterministically from the session's own history, so a student trained on
the data learns to use the notes it is served with.

Live, a consolidator model (DeepSeek) rewrites a small JSON object after
each turn. Here nothing is asked of a model: the generator owns the state of
a game it sets up, and reads the rest off the viewer's lines.

  game         a session may carry one game, planned from the session's seed
               alone (plan_game, its own generator, so the draws of the
               session itself do not move): "number" (the character secretly
               picks 1-50, the viewer guesses), "twenty" (20 questions about
               a secret thing of hers), "chain" (an English word chain),
               "rather" (would you rather, several rounds). It starts on
               turn 2 or 3 and lasts 4 to 8 turns. The simulated viewer is
               told what to do each turn (viewer_hint); the notes say what
               she needs to answer: her secret, the guesses so far and
               whether each was too small or too big, the questions asked,
               the words used, the picks. The two games with a secret of
               hers only run when notes are on: without them she could not
               keep it.
  corrections  a line that corrects her (flow.checks.corrections: "not X, it
               is Y") adds "Y, never X".
  facts        a viewer who says their name ("my name is Mika", 我叫小明,
               名前は…です) adds "viewer's name is Mika".

What the notes say at turn t is what was known after turn t-1 (live, the
consolidator runs after a turn and lands before the next), except that a
game's setup (her secret) is there from its first turn: she has to be able to
pick it. The block is rendered with flow.memory.Notes.fit / render, the live
renderer, and is not part of the history: the record keeps it in "tail"
beside the plain "user_content" (flow.dataset).
"""

from __future__ import annotations

import os
import random
import re
import sys
import zlib

try:
    from . import memory as mem
except ImportError:                       # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from flow import memory as mem

GAMES = ("number", "twenty", "chain", "rather")
NEEDS_NOTES = ("number", "twenty")        # a secret of hers
NUMBER_RANGE = (1, 50)
SECRETS = ["a lighthouse", "a red umbrella", "a violin", "a pineapple",
           "a bicycle", "a snowman", "a teapot", "a kite", "a library book",
           "a hot air balloon", "a toothbrush", "a garden gnome"]
QUESTIONS = 20
SEEN_GUESSES = 5                          # guesses kept in the note


class Plan(dict):
    """{kind, start, length, secret}: kind in GAMES, start the first turn
    of the game (1-based), length its turns, secret the number / thing
    (number, twenty)."""

    def active(self, turn) -> bool:
        return self["start"] <= turn < self["start"] + self["length"]

    @property
    def end(self) -> int:
        return self["start"] + self["length"]


def plan_game(seed, turns, share, notes_on, kinds=None) -> Plan | None:
    """The game of a session, or None: with chance `share`, from the
    session's seed alone (so nothing else of the session moves). A session
    shorter than 6 turns has none; without notes only the games that need
    no secret of hers are drawn."""
    if share <= 0 or turns < 6:
        return None
    r = random.Random(zlib.crc32(f"{seed}:game".encode()))
    if r.random() >= share:
        return None
    pool = [k for k in (kinds or GAMES)
            if notes_on or k not in NEEDS_NOTES]
    if not pool:
        return None
    kind = r.choice(pool)
    start = r.choice((2, 3))
    length = min(r.randint(4, 8), turns - start + 1)
    secret = (r.randint(*NUMBER_RANGE) if kind == "number"
              else r.choice(SECRETS) if kind == "twenty" else None)
    return Plan(kind=kind, start=start, length=length, secret=secret)


_INT = re.compile(r"(?<![\d.])(\d{1,3})(?![\d])")
_WORD = re.compile(r"[A-Za-z]+")
_NAMES = [re.compile(p, re.I) for p in (
    r"\bmy\s+name\s+is\s+([A-Z][\w'-]{1,15})", r"\bcall\s+me\s+([A-Z][\w'-]{1,15})",
    r"\b(?:I'm|I\s+am)\s+([A-Z][a-z'-]{1,15})\b(?!\s+(?:so|very|just|not|"
    r"really|a|an|the|going|feeling|sorry|tired|happy|sad|fine|good|back|"
    r"here))")] + [re.compile(p) for p in (
        r"我叫\s*([^\s，。,.!?！？]{1,6})", r"名前は\s*([^\s、。,.!?！？です]{1,8})",
        r"(?:私|僕|ぼく|俺)は\s*([^\s、。,.!?！？は]{1,8})\s*(?:です|といいます)")]


def guess_of(line):
    """The whole number a viewer guesses in a line (the first within the
    game's range), or None."""
    for m in _INT.finditer(line or ""):
        v = int(m.group(1))
        if NUMBER_RANGE[0] <= v <= NUMBER_RANGE[1]:
            return v
    return None


def word_of(line):
    """The word a viewer plays in a word chain: the last Latin word of a
    line of at most 3 words, lower-cased; None for a longer line."""
    ws = _WORD.findall(line or "")
    if not ws or len(re.findall(r"\S+", line or "")) > 3:
        return None
    return ws[-1].lower()


def name_of(line):
    for rx in _NAMES:
        m = rx.search(line or "")
        if m:
            return m.group(1).strip("'-")
    return None


class Keeper:
    """The notes of one session (module docstring)."""

    def __init__(self, plan=None, chars=mem.CHARS):
        self.plan, self.chars = plan, int(chars)
        self.notes = mem.Notes()
        self.guesses, self.words, self.picks = [], [], []
        self.questions = 0

    def game_line(self, turn):
        """The game entry now, for turn `turn`, or None."""
        p = self.plan
        if p is None or not p.active(turn):
            return None
        k = p["kind"]
        if k == "number":
            lo, hi = NUMBER_RANGE
            s = f"number guess {lo}-{hi}, her secret is {p['secret']}"
            if self.guesses:
                s += ", guesses: " + ", ".join(
                    f"{g} {v}" for g, v in self.guesses[-SEEN_GUESSES:])
            return s
        if k == "twenty":
            return (f"20 questions, her secret is {p['secret']}, "
                    f"{self.questions} of {QUESTIONS} questions asked")
        if k == "chain":
            s = "word chain, each word starts with the last letter of " \
                "the one before"
            if self.words:
                s += ", used: " + ", ".join(self.words[-8:])
            return s
        rnd = len(self.picks)
        s = f"would you rather, round {rnd + 1}"
        if self.picks:
            s += ", viewer picked: " + " / ".join(self.picks[-3:])
        return s

    def block(self, turn) -> tuple:
        """(the "(stream notes: ...)" block or "", the Notes object) for
        turn `turn` of the session; the Notes are the live renderer's, fitted
        to its character budget."""
        n = self.notes.copy()
        g = self.game_line(turn)
        n.game = []
        if g:
            n.add("game", g, self.chars)
        n.fit(self.chars)
        return n.render(), n

    def observe(self, turn, line, spoken="", prev_spoken=""):
        """Take the turn in: what the viewer wrote on `turn` and what she
        said. Nothing is read from quiet-chat turns (the caller skips them)."""
        try:
            from . import checks as ck
        except ImportError:
            from flow import checks as ck
        for x, y in ck.corrections(line, prev_spoken):
            if y:
                self.notes.add("corrections", f"{y}, never {x}", self.chars)
        nm = name_of(line)
        if nm:
            self.notes.add("facts", f"viewer's name is {nm}", self.chars)
        p = self.plan
        if p is None or not p.active(turn) or turn == p["start"]:
            return
        k = p["kind"]
        if k == "number":
            g = guess_of(line)
            if g is not None:
                s = p["secret"]
                self.guesses.append((g, "correct" if g == s else
                                     "too small" if g < s else "too big"))
        elif k == "twenty":
            self.questions += 1
        elif k == "chain":
            w = word_of(line)
            if w and w not in self.words:
                self.words.append(w)
        else:
            t = mem.clean(line)[:28].rstrip()
            if t:
                self.picks.append(t)


def viewer_hint(plan, turn, keeper) -> str | None:
    """What the simulated viewer does on a game turn (the topic's hint), or
    None when no game is on."""
    if plan is None or not plan.active(turn):
        return None
    k, first = plan["kind"], turn == plan["start"]
    lo, hi = NUMBER_RANGE
    if k == "number":
        if first:
            return (f"propose a number guessing game: the character secretly "
                    f"picks a whole number from {lo} to {hi}, you will guess, "
                    "and she says higher or lower; ask her to say when she "
                    "is ready")
        seen = ", ".join(str(g) for g, _v in keeper.guesses) or "none yet"
        return (f"you are playing the number guessing game (a whole number "
                f"from {lo} to {hi}): guess one new whole number, written "
                "with digits, using her higher / lower answers so far (your "
                f"guesses: {seen}); add a few words of reaction at most")
    if k == "twenty":
        if first:
            return ("propose 20 questions: the character secretly thinks of "
                    "an everyday object and you ask yes / no questions")
        if keeper.questions >= QUESTIONS - 1:
            return ("you are playing 20 questions and this is your last "
                    "turn: make your final guess of what the object is")
        return (f"you are playing 20 questions (question "
                f"{keeper.questions + 1} of {QUESTIONS}): ask one new "
                "yes / no question about her secret object, using her "
                "earlier answers")
    if k == "chain":
        if first:
            return ("propose an English word chain: each word must start "
                    "with the last letter of the previous word and no word "
                    "repeats; you go first with a single word. Write in "
                    "English")
        return ("you are playing the English word chain: write only your "
                "next single word, starting with the last letter of her "
                "last word, not one used before. Write in English")
    if first:
        return ("propose would-you-rather: ask her a silly would-you-rather "
                "question with two options")
    return ("you are playing would-you-rather: pick one of her two options "
            "with a short reason, then ask her your own would-you-rather "
            "question")


def lang_for(plan, turn):
    """The language a game turn is played in, or None to leave it to the
    session (the word chain is English)."""
    if plan is not None and plan.active(turn) and plan["kind"] == "chain":
        return "en"
    return None
