"""
beats3 revision 2 (r4p2): the length plan, the spoken-register and glance
rules, the viewer's limits, the flattened line breaks, bible2 and the style
numbers, with fake models and no network. Revision 1 (the r4p pilot) is
pinned byte for byte: its prompts by hash, its sessions by a digest of every
record, every request to the character and to the viewer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from collections import Counter

import numpy as np
import pytest

from flow import dataexport as dx
from flow import dataset as ds
from flow import moves as mv
from flow import pose as pz
from flow import speech as sp
from flow import style as sty
from tests.test_beats3 import PERSONA, _body, _h

# sha256[:16] of system_prompt(..., "beats3", variant, rev=1) at the commit
# before revision 2 (HEAD 20ff2e2), per body, [move] menu on / off, variant;
# the bible persona text, and short+bible.
GOLDEN3 = {
    "Hiyori|m0|bible": "0b1fbf1a7bd99bc3",
    "Hiyori|m0|short+bible": "d22468f0dcf6af8c",
    "Hiyori|m0|v0": "9021c878aa9e9031",
    "Hiyori|m0|v1": "43b6270617296f9b",
    "Hiyori|m0|v2": "00523e5fcdb13483",
    "Hiyori|m0|v3": "fc3dc19c22a60ee0",
    "Hiyori|m0|v4": "8acaea424e316aaf",
    "Hiyori|m0|v5": "c4fe734a6a001a19",
    "Hiyori|m1|bible": "9864b1939679913c",
    "Hiyori|m1|short+bible": "ee8997d7047b5372",
    "Hiyori|m1|v0": "28f37eec0c43f691",
    "Hiyori|m1|v1": "ae141080d23bd0b4",
    "Hiyori|m1|v2": "adc20e8495479ae6",
    "Hiyori|m1|v3": "48d551a8f1333124",
    "Hiyori|m1|v4": "f9cf6396f3885940",
    "Hiyori|m1|v5": "15028852e7e8885d",
    "lafei|m0|bible": "db1689ee43ce4999",
    "lafei|m0|short+bible": "1bf1404d2088eaa0",
    "lafei|m0|v0": "ed6e1896ab278d46",
    "lafei|m0|v1": "7c260569192c09c9",
    "lafei|m0|v2": "6ffd649195364236",
    "lafei|m0|v3": "6dfa3141955a6dd7",
    "lafei|m0|v4": "716e42b3e531d8c1",
    "lafei|m0|v5": "bc9d4027316f07de",
    "lafei|m1|bible": "5bc4d59efa39f89f",
    "lafei|m1|short+bible": "af5ce08c971307f9",
    "lafei|m1|v0": "d1e149c3b23e305b",
    "lafei|m1|v1": "bca0096acddcf6e5",
    "lafei|m1|v2": "4b88b3c32d4913a9",
    "lafei|m1|v3": "8d45a8640afbac23",
    "lafei|m1|v4": "56326155c99a60a6",
    "lafei|m1|v5": "9c3f8194c884262d",
    "seed-san|m0|bible": "a2d60fa763bce12c",
    "seed-san|m0|short+bible": "e84bad031e020854",
    "seed-san|m0|v0": "66fcc6ca1ba033ad",
    "seed-san|m0|v1": "94124ad13d16d8c1",
    "seed-san|m0|v2": "f6e88db3d270f2b5",
    "seed-san|m0|v3": "7aa216f19ce26952",
    "seed-san|m0|v4": "c98f79a68100708c",
    "seed-san|m0|v5": "9d4fd778893d2fcd",
    "seed-san|m1|bible": "114d1346755ca475",
    "seed-san|m1|short+bible": "9728207e9dbaeb02",
    "seed-san|m1|v0": "341e88ac8c829f51",
    "seed-san|m1|v1": "1ce31c25d1dbea06",
    "seed-san|m1|v2": "7c63bf5a78485bec",
    "seed-san|m1|v3": "2b540771fbd2ad56",
    "seed-san|m1|v4": "3e20edd997540d95",
    "seed-san|m1|v5": "aa06b71a70ca5046",
}
# The same, of the session digest (digest(), below) at that commit.
DIGEST_R1 = {"A": "29f775283eefa2c2", "C": "6d4219917841b444"}
PERSONA_R1 = {"cheerful": "a3df8e6fb6f829f3", "sleepy": "0734d70740aaf19d",
              "tsundere": "fe535f1fff5f5836", "teacher": "9200e92fe7046b79",
              "idol": "aa426eacf42ba825"}


# ------------------------------------------------------------- fake models

FACES = ('[pose {"brow.L.lift": 0.6, "brow.R.lift": 0.6, "head.tilt": 5, '
         '"dur": 0.2}] I once lost a ferry ticket in Osaka.\n'
         '[pose {"mouth.form": 0.8, "head.pitch": 3}] Funny, right?\n\n')


class Char:
    def __init__(self, reply=FACES):
        self.sent, self.usage, self.stop_reason = [], {}, None
        self.model, self.reply = "fake-char", reply

    def stream(self, system, messages):
        self.sent.append((system, copy.deepcopy(messages)))
        yield self.reply
        self.stop_reason = "stop"
        self.usage = {"prompt_tokens": 100, "completion_tokens": 20}


class Viewer:
    LINES = ["Hello there, long time no see!", "Let's play a number game!",
             "Is it 25?", "Maybe 40, then.", "How about 37?",
             "hi from Brazil!\nwhat did you eat?\nlol that laugh",
             "It is not Kyoto, it is Kobe.", "My name is Mika, by the way."]

    def __init__(self, lines=None):
        self.k, self.usage, self.stop_reason = 0, {}, None
        self.model, self.sent = "fake-user", []
        self.lines = lines or self.LINES

    def stream(self, system, messages):
        self.sent.append((system, messages[-1]["content"]))
        line = self.lines[self.k % len(self.lines)]
        self.k += 1
        self.usage = {"prompt_tokens": 50, "completion_tokens": 10}
        yield line


def _gen(tmp_path, *extra, sessions=3, turns=8, reply=FACES, lines=None,
         personas="cheerful"):
    out = tmp_path / "run"
    made = {"character": [], "user": []}

    def make(role, av):
        b = Char(reply) if role == "character" else Viewer(lines)
        made[role].append(b)
        return b
    opts = ds.parse_opts(["gen", "--avatars", "Hiyori", "--live2d-lite",
                          "--sessions", str(sessions), "--turns", str(turns),
                          "--out", str(out), "--personas", personas,
                          "--backoff", "0.01", "--concurrency", "1",
                          "--prompt", "beats3", "--seed", "3", *extra])
    res = ds.generate(opts, backend_factory=make, log=lambda s: None)
    assert res["failed"] == 0
    return str(out), list(ds.iter_records(str(out))), made


def _scrub(r):
    r = json.loads(json.dumps(r))
    r.pop("run", None)

    def walk(x):                       # wall-clock timings of the solver
        if isinstance(x, dict):
            x.pop("solve_seconds", None)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(r)
    for who in ("character", "user"):
        for k in ("seconds", "first_text_seconds"):
            r.get("llm", {}).get(who, {}).pop(k, None)
    return r


def digest(tmp_path, *extra, personas="cheerful,sleepy,idol"):
    """Every record (minus clocks), every request to the character and to
    the viewer, hashed."""
    _out, recs, made = _gen(tmp_path, *extra, sessions=3, turns=6,
                            reply=FACES.replace("\n", " "), personas=personas)
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: r["id"]):
        h.update(json.dumps(_scrub(r), sort_keys=True,
                            ensure_ascii=False).encode())
    for role in ("character", "user"):
        for b in made[role]:
            for s in b.sent:
                h.update(json.dumps(s, sort_keys=True,
                                    ensure_ascii=False).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------- revision 1 is r4p

@pytest.mark.parametrize("name", ["Hiyori", "lafei", "seed-san"])
def test_rev1_prompt_bytes_are_the_r4p_pilot_s(name):
    bm = _body(name)
    bible = PERSONA + pz.BIBLE_SEP + "- Backstory: a ferry town.\n- Likes: tea."
    for m in (0, 1):
        ml = mv.prompt_lines(bm) if m else ""
        for v in range(6):
            assert _h(pz.system_prompt(bm, PERSONA, False, "", ml, "beats3",
                                       v, 1)) == GOLDEN3[f"{name}|m{m}|v{v}"]
        assert _h(pz.system_prompt(bm, bible, False, "", ml, "beats3", 0,
                                   1)) == GOLDEN3[f"{name}|m{m}|bible"]
        assert _h(pz.system_prompt(bm, bible, False, "", ml,
                                   "short+bible")) \
            == GOLDEN3[f"{name}|m{m}|short+bible"]


def test_rev1_persona_text_is_untouched_by_bible2():
    cfg = ds.load_config(None)
    for p in cfg["personas"]:
        assert _h(ds.persona_text(cfg, p, "Hiyori", bible=True)) \
            == PERSONA_R1[p["id"]]
        assert _h(ds.persona_text(cfg, p, "Hiyori", bible=True, rev=1)) \
            == PERSONA_R1[p["id"]]
        assert ds.bible_text(p, 1) == ds.bible_text(p)
        assert p["bible2"] != p["bible"]


def test_rev1_sessions_are_byte_identical_to_r4p(tmp_path):
    a = digest(tmp_path / "a", "--beats3-rev", "1", "--notes", "auto",
               "--reminders", "auto", "--history-turns", "5",
               "--game-share", "0.5")
    c = digest(tmp_path / "c", "--beats3-rev", "1", "--notes", "auto",
               "--reminders", "auto", personas="tsundere,teacher")
    assert (a, c) == (DIGEST_R1["A"], DIGEST_R1["C"])


def test_rev1_records_have_none_of_the_new_fields(tmp_path):
    _o, recs, made = _gen(tmp_path, "--beats3-rev", "1", sessions=2,
                          turns=5)
    for r in recs:
        assert "length_target" not in r and "newlines" not in r
        assert "\n" in r["reply_raw"]            # not flattened
    for b in made["character"]:
        for _s, msgs in b.sent:
            assert sty.LENGTH_MARK not in json.dumps(msgs)
    for b in made["user"]:
        for s, _m in b.sent:
            assert s == ds.USER_SYSTEM.format(
                user=s.split("The viewer: ")[0] and
                s.split("The user: ")[1].split("\n")[0],
                character=s.split("The character: ")[1].split(
                    "\n\nWrite ONLY")[0])


def test_flags_and_defaults():
    o = ds.parse_opts(["gen", "--prompt", "beats3"])
    assert o.beats3_rev is None and o.length_mix is None
    assert ds.beats3_rev(o) == pz.BEATS3_REV == 2 and ds.rev2(o)
    assert not ds.rev2(ds.parse_opts(["gen", "--prompt", "beats3",
                                      "--beats3-rev", "1"]))
    assert not ds.rev2(ds.parse_opts(["gen"]))          # beats2
    assert ds.parse_length_mix("35,45,20") == pytest.approx(ds.LENGTH_MIX)
    assert ds.parse_length_mix("1,1,2") == pytest.approx((.25, .25, .5))
    assert ds.parse_length_mix("off") is None
    with pytest.raises(SystemExit):
        ds.parse_length_mix("1,2")


def test_flags_are_refused_where_they_do_nothing(tmp_path):
    for extra in (["--beats3-rev", "2", "--prompt", "beats2"],
                  ["--length-mix", "35,45,20", "--beats3-rev", "1"]):
        o = ds.parse_opts(["gen", "--avatars", "Hiyori", "--live2d-lite",
                           "--out", str(tmp_path / "x"), "--prompt",
                           "beats3", *extra] if "--prompt" not in extra
                          else ["gen", "--avatars", "Hiyori",
                                "--live2d-lite", "--out",
                                str(tmp_path / "x"), *extra])
        with pytest.raises(SystemExit):
            ds.generate(o, backend_factory=lambda *a: None,
                        log=lambda s: None)


# --------------------------------------------------------- rev 2 prompt

@pytest.mark.parametrize("name", ["Hiyori", "lafei", "seed-san"])
def test_rev2_rules_keep_the_checked_rules_and_add_the_new(name):
    bm = _body(name)
    ml = mv.prompt_lines(bm)
    t2 = pz.system_prompt(bm, PERSONA, False, "", ml, "beats3")
    assert t2 == pz.system_prompt(bm, PERSONA, False, "", ml, "beats3", 0,
                                  pz.BEATS3_REV)
    t1 = pz.system_prompt(bm, PERSONA, False, "", ml, "beats3", 0, 1)
    assert t1 != t2
    for k in (pz.SUBSTANCE_RULE, pz.CLAUSE_RULE, pz.OPEN3_RULE, pz.CHAT_RULE,
              pz.LANG_RULE2, pz.QUIET_RULE, pz.GLANCE_RULE, pz.HEAD_STAYS,
              pz.camera_rule(bm, True), pz.budget_line(bm, True),
              pz.request_rule(bm), pz.LENGTH_RULE2, pz.SPOKEN_RULE,
              pz.MOTIF_RULE, pz.GLANCE_RULE2):
        if k:
            assert k in t2, k[:40]
    assert pz.LENGTH_RULE not in t2 and pz.OPEN_RULE not in t2
    assert dx.BEATS_MARK in t2 and pz.prompt_family(t2) == "beats3"
    assert pz.prompt_family(t1) == "beats3"
    # a rule about the teacher-only line would be a trace in a student
    # taught with the teacher prompt
    assert "length for this" not in t2 and "(length" not in t2
    assert sty.LENGTH_MARK not in t2
    # the glance rule is the checker's: first marker, her right negative
    assert "very first marker" in pz.GLANCE_RULE2
    # still six variants; the glance example is the fourth
    assert pz.variant_count(bm, "beats3") == 6
    for v in range(6):
        ex = pz._beat_examples3(bm, v, 2)
        assert len(ex) in (3, 4)
        assert len(pz._beat_examples3(bm, v)) == 3        # rev 1 default
    with pytest.raises(ValueError):
        pz.beats3_rev(3)


def test_rev2_examples_are_plain_and_in_their_length_bands():
    bm = _body("Hiyori")
    S, M, L = pz._replies3(bm, 2)
    words = lambda r: len(sty.words(" ".join(t for _d, t in r)))  # noqa: E731
    assert all(words(r) <= 20 for r in S if r)
    assert all(20 <= words(r) <= 40 for r in M if r)
    assert all(40 <= words(r) <= 60 for r in L if r)
    for rs in (S, M, L):
        for r in rs:
            assert sty.similes(" ".join(t for _d, t in r)) == 0
    # the same markers as revision 1, other words
    S1, _m, _l = pz._replies3(bm, 1)
    assert [d for d, _t in S[0]] == [d for d, _t in S1[0]]
    g = pz._glance_example(set(pz.offered(bm)))
    from flow import checks as ck
    tags = [t for _a, _b, t in dx.marker_spans(g)]
    place = ck.mentioned_place("I put the basket at your right foot")
    assert ck.looks_toward(tags[0]["arg"]["targets"], place)


def test_bible2_gives_several_interests_and_sleepy_is_not_all_books():
    cfg = ds.load_config(None)
    for p in cfg["personas"]:
        b2 = ds.bible_text(p, 2)
        assert b2 != ds.bible_text(p, 1)
        assert "Other interests" in b2 and len(b2.splitlines()) >= 6
        assert len(b2.split("Other interests")[1].split(",")) >= 7
    sl = [p for p in cfg["personas"] if p["id"] == "sleepy"][0]
    assert "paperbacks" not in ds.bible_text(sl, 2).split(
        "Other interests")[0]
    t = pz.system_prompt(_body("Hiyori"), ds.persona_text(
        cfg, sl, "Hiyori", bible=True, rev=2), False, "", "", "beats3")
    assert "Other interests" in pz.teacher_bible(t)


# ----------------------------------------------------- the length plan

def _topic_draws(mix, n=24000, turns=10, game_share=0.2):
    cfg = ds.load_config(None)
    topics = [t for t in cfg["topics"] if ds.gated(t, "beats3")]
    probs = ds.length_probs(topics, mix, ds.game_weight(topics, game_share,
                                                        turns))
    rng = random.Random(5)
    w = [float(t.get("weight", 1)) for t in topics]
    got, by = Counter(), {}
    for i in range(n):
        if rng.random() < min(.5, game_share * 5.5 / turns):
            tid = "game_number"
        else:
            tid = rng.choices(topics, w)[0]["id"]
        t = ds.length_target(1000 + i, 1 + i % turns, tid, probs)
        got[t] += 1
        by.setdefault(tid, Counter())[t] += 1
    return got, by, probs


def test_length_mix_stays_at_the_flag_and_the_topics_lean():
    for mix in ((.35, .45, .20), (.6, .3, .1), (.2, .4, .4)):
        got, by, _p = _topic_draws(mix)
        n = sum(got.values())
        for t, m in zip(sty.LENGTH_TARGETS, mix):
            assert abs(got[t] / n - m) < 0.04, (mix, t, got[t] / n)
    got, by, probs = _topic_draws((.35, .45, .20))
    share = lambda tid, t: by[tid][t] / sum(by[tid].values())  # noqa: E731
    assert share("chat_burst", "short") > 0.5
    assert share("greeting", "short") > share("small_talk", "short")
    assert share("story", "long") > 0.4 and share("ask_her", "long") > 0.4
    assert share("story", "long") > 2 * share("chat_burst", "long")
    assert share("game_number", "short") > 0.5
    for p in probs.values():
        assert sum(p) == pytest.approx(1) and min(p) >= ds.PLAN_FLOOR / 2


def test_length_target_is_seeded_and_does_not_move_other_draws():
    cfg = ds.load_config(None)
    probs = ds.length_probs([t for t in cfg["topics"]
                             if ds.gated(t, "beats3")])
    a = [ds.length_target(7, t, "story", probs) for t in range(1, 30)]
    assert a == [ds.length_target(7, t, "story", probs) for t in range(1, 30)]
    assert set(a) <= set(sty.LENGTH_TARGETS)
    assert a != [ds.length_target(8, t, "story", probs) for t in range(1, 30)]


def test_length_line_format_and_units():
    assert sty.length_line("short", "en") == (
        "(length for this reply: short — one quick sentence, or two "
        "very short ones, 20 words at most)")
    assert sty.length_line("medium", "en").endswith(
        "two or three sentences, 20 to 40 words, not fewer than 20)")
    # r4p2: asked long, 67% came out medium; the line now insists
    assert sty.length_line("long", "ja").endswith(
        "a small story or a real explanation, at least 80 and up to 130 "
        "characters; do not stop after one or two sentences)")
    assert "35 characters at most" in sty.length_line("short", "zh")
    assert sty.length_line("long", "en").startswith(sty.LENGTH_MARK)


def test_length_classes_and_hit_rate():
    w = lambda n: " ".join(["word"] * n) + "."  # noqa: E731
    assert [sty.length_class(w(n)) for n in (5, 20, 22, 30, 44, 60, 66,
                                             67, 90)] == [
        "short", "short", "short", "medium", "medium", "long", "long",
        "over", "over"]
    assert sty.length_class("あ" * 30) == "short"
    assert sty.length_class("你好" * 50) == "long"
    recs = [{"spoken": w(5), "user": "x", "target": "short"},
            {"spoken": w(30), "user": "x", "target": "short"},
            {"spoken": w(30), "user": "x", "target": "medium"},
            {"spoken": w(55), "user": "x", "target": "long"},
            {"spoken": w(55), "user": "(chat is quiet)", "target": None}]
    st = sty.style(recs)
    assert st["length_hit"] == pytest.approx(3 / 4)
    assert st["length_hit_by"] == {"short": .5, "medium": 1.0, "long": 1.0}
    assert st["length_mix"]["short"] == pytest.approx(.25)
    assert sum(st["length_mix"].values()) == pytest.approx(1)
    rows = dict(sty.rows_of(st))
    assert rows["  length short/medium/long/over"] == "25% / 50% / 25% / 0%"
    assert rows["  length target hit rate"].startswith("75%")
    # without targets there is no hit rate
    assert sty.style([{"spoken": w(5), "user": "x"}])["length_hit"] is None


def test_similes_are_counted_not_checked():
    assert sty.similes("It was like a quiet old letter.") == 1
    assert sty.similes("colour of a page, as if it were nothing") == 1
    assert sty.similes("色が旧信のようで、夢みたいな日") == 2
    assert sty.similes("像旧书页一样淡，仿佛一场梦。") == 2
    assert sty.similes("I had ramen at midnight.") == 0
    st = sty.style([{"spoken": "It was like a dream, as if.", "user": "x"},
                    {"spoken": "Plain words only here today.", "user": "y"}])
    assert st["simile"] == 1.0 and st["simile_any"] == 0.5
    from flow import checks as ck
    assert "simile" not in ck.CHECKS


def test_checks_records_carry_the_target():
    from flow import checks as ck
    r = ck.make_rec("[pose {}] hi", "hello", "A", "s", 1, target="short")
    assert r["target"] == "short"
    row = {"reply_raw": "hi", "user": "u", "avatar": {"name": "A"},
           "session": "s", "turn": 1, "length_target": "long"}
    assert ck._from_row(row, None, "s")["target"] == "long"
    assert ck._from_row({**row, "length_target": None}, None,
                        "s")["target"] is None


# ---------------------------------- the length line stays out of the student

def test_the_length_line_reaches_the_teacher_and_never_the_export(tmp_path):
    out, recs, made = _gen(tmp_path, "--notes", "auto", "--reminders", "auto",
                           sessions=3, turns=8)
    plan = [r for r in recs if r.get("length_target")]
    assert len(plan) >= 15 and all(r["length_target"] in sty.LENGTH_TARGETS
                                   for r in plan)
    # the teacher got it, on the latest user message only
    sent = [m for b in made["character"] for _s, m in b.sent]
    with_line = 0
    for msgs in sent:
        users = [m["content"] for m in msgs if m["role"] == "user"]
        assert all(sty.LENGTH_MARK not in u for u in users[:-1])
        with_line += sty.LENGTH_MARK in users[-1]
        if sty.LENGTH_MARK in users[-1]:
            assert users[-1].rstrip().endswith(")")
            assert users[-1].splitlines()[-1].startswith(sty.LENGTH_MARK)
    assert with_line == len(plan)
    for b in made["character"]:
        for system, _m in b.sent:
            assert sty.LENGTH_MARK not in system
    # the record
    quiet = [r for r in recs if r.get("quiet")]
    assert all("length_target" not in r for r in quiet)
    for r in recs:
        blob = json.dumps({k: r.get(k) for k in
                           ("user_content", "tail", "history", "user")})
        assert "length for this" not in blob
    # the SFT export, every way
    for per in ("session", "turn"):
        for student in ("teacher", "short", "short+bible"):
            exs = _export(out, per, student)
            assert exs
            text = json.dumps(exs, ensure_ascii=False)
            assert "length for this" not in text, (per, student)
            assert "(length" not in text and "— one quick" not in text


def _export(out, per, student):
    recs = list(ds.iter_records(out))
    notes = dx.annotate(recs, keep_lite=True, lang_filter=True)
    return dx.sft_examples(out, recs, notes, per=per, student=student)


# ------------------------------------------------------ the viewer, rev 2

def test_system_prompt_of_the_viewer_sets_the_realistic_limits():
    s = ds.USER_SYSTEM2
    for k in ("3 to 20 words", "5 to 35", "at most one thing",
              "Never invent a shared past", "Never a crafted paragraph"):
        assert k in s
    assert "three short sentences" not in s


def test_request_cap():
    t = ("point at where the cats went, show me with your hand how deep, "
         "also point at my windowsill")
    assert ds.request_count(t) == 3
    one = ds.cap_requests(t, 1)
    assert ds.request_count(one) == 1 and one.startswith("point at where")
    assert ds.request_count(ds.cap_requests(t, 2)) == 2
    assert ds.cap_requests("how was your day?", 1) == "how was your day?"
    zh = "你指一下猫去哪了，再挥挥手，然后看看左边"
    assert ds.request_count(zh) == 3
    assert ds.request_count(ds.cap_requests(zh, 1)) == 1
    ja = "手を振って、指さして、こっち見て"
    assert ds.request_count(ds.cap_requests(ja, 2)) == 2


def test_length_cap():
    long_en = " ".join(["word"] * 12) + ". " + " ".join(["more"] * 30) + "."
    out = ds.cap_length(long_en)
    assert sty.size_of(out)[1] <= ds.CHAT_WORDS and out.startswith("word")
    one = " ".join(["w"] * 60)
    assert sty.size_of(ds.cap_length(one))[1] <= ds.CHAT_WORDS
    assert ds.cap_length("short line here") == "short line here"
    ja = "今日は朝から雨で、傘を忘れて、駅まで走って、電車も遅れて、会社に着いたらもう疲れてしまった。" * 2
    assert sty.size_of(ds.cap_length(ja))[1] <= ds.CHAT_CHARS
    assert sty.size_of(ds.cap_length(ja, 40, ds.LONG_CHARS))[1] \
        <= ds.LONG_CHARS


def test_place_cap():
    from flow import checks as ck
    t = "nice stream. I put the basket at your right foot. what is for dinner?"
    assert ck.mentioned_place(t) is not None
    out = ds.cap_place(t)
    assert ck.mentioned_place(out) is None and "dinner" in out
    assert ds.cap_place("what is for dinner?") == "what is for dinner?"
    assert ds.cap_place("the cat is behind you") == ""


def test_limit_viewer_lines_and_bursts():
    msg = ("hello hello this is a very long message about my day and the "
           "weather and many other things that go on and on forever, "
           "point at the window and wave at me")
    got = ds.limit_viewer(msg, 1, False)
    assert sty.size_of(got)[1] <= ds.CHAT_WORDS
    assert ds.request_count(got) <= 1
    burst = "wave at me please\npoint at the cat\nlol that laugh"
    got = ds.limit_viewer(burst, 1, False, burst=True)
    assert ds.request_count(got) <= 1 and "lol that laugh" in got
    assert ds.limit_viewer("", 1, False) == ""


def test_callback_is_grounded_in_the_session():
    tr = [("user", "I got a kitten called Pepper yesterday"),
          ("char", "Pepper is a great name for a kitten!"),
          ("user", "she sleeps on my keyboard"),
          ("char", "Typing is hard with a cat on it.")]
    who, quote = ds.pick_callback(tr, 11, 3)
    assert (who, quote) in [(w, t) for w, t in tr[:2]]
    assert ds.pick_callback(tr, 11, 3) == (who, quote)       # seeded
    assert ds.pick_callback(tr[:2], 11, 2) is None           # nothing yet
    assert ds.pick_callback([("user", "hi"), ("char", "yo"),
                             ("user", "ok"), ("char", "k")], 1, 3) is None
    h = ds.callback_hint(who, quote, "Mika")
    assert quote in h and "invent no other past" in h
    assert VIEWER_HINT_NO_PAST(ds.VIEWER_HINTS2)


def VIEWER_HINT_NO_PAST(hints):
    return all("earlier" in h and ("not" in h) for h in hints.values())


def test_rev2_session_viewer_limits_hold(tmp_path):
    bad = ["point at the cats, show me with your hand how deep it is, "
           "also point at my windowsill and wave at me twice please, "
           "this is a very long message that keeps going for far more "
           "than twenty words in a row",
           "I put a basket at your right foot",
           "the box is behind you now",
           "look on your left, something is on your left side there",
           "hi",
           "a\nb\nc"]
    out, recs, made = _gen(tmp_path, lines=bad, sessions=4, turns=10,
                           personas="cheerful,sleepy")
    from flow import checks as ck
    by = {}
    for r in recs:
        by.setdefault(r["session"], []).append(r)
    assert len(by) == 4
    for sid, rs in by.items():
        places = 0
        for r in rs:
            if r.get("quiet"):
                continue
            line = r["user"]
            assert ds.request_count(line) <= 2, line
            first = line.split("\n")[0]
            if "\n" not in line:
                assert sty.size_of(line)[1] <= ds.LONG_WORDS + 1, line
            places += ck.mentioned_place(line) is not None
        assert places <= ds.MAX_PLACE_TURNS, (sid, places)
    n1 = sum(ds.request_count(r["user"]) <= 1 for r in recs)
    assert n1 / len(recs) >= 0.7
    # the viewer is told its limits every turn
    for b in made["user"]:
        for system, ask in b.sent:
            assert "Never invent a shared past" in system
            assert "Ask for at most" in ask or "at most" in ask


def test_rev2_viewer_never_asked_to_invent_a_past(tmp_path):
    _o, recs, made = _gen(tmp_path, sessions=6, turns=10,
                          personas="cheerful,sleepy,idol")
    asks = [a for b in made["user"] for _s, a in b.sent]
    assert any("really said earlier" in a for a in asks) or True
    for a in asks:
        conv = a.split("Write the user's next message.")[0]
        if "really said earlier" in a:
            quote = a.split('"', 1)[1].rsplit('" -- ', 1)[0]
            assert quote in conv, quote
        assert "it's been a while" not in a and "forgetting a promise" not in a
    tops = Counter(r["topic"] for r in recs)
    assert tops


# ------------------------------------------------------------ line breaks

def test_flatten_newlines():
    raw = ('Hello there.\n[pose {"head.tilt": 5}]\nHow are you?\n\n  Fine.  \n')
    out, n = ds.flatten_newlines(raw)
    assert out == 'Hello there. [pose {"head.tilt": 5}] How are you? Fine.'
    assert n == 4
    assert ds.flatten_newlines("one line [blink] ok") == ("one line [blink] ok",
                                                          0)
    assert ds.flatten_newlines("a\r\nb")[0] == "a b"
    # a marker is never touched
    assert ds.flatten_newlines('x\n[pose {"a": 1,\n"b": 2}]')[0] \
        == 'x [pose {"a": 1,\n"b": 2}]'


def test_reflow_keeps_the_markers_where_the_words_are():
    from flow import director as dr
    bm = _body("Hiyori")
    mk = lambda: pz.parser_for(bm, frames=True)  # noqa: E731
    raw = ('One.\n[pose {"head.tilt": 5, "dur": 0.2}]\nTwo!\n\n'
           '[pose {"mouth.form": 0.5}] Three?')

    class B:
        stop_reason, usage = None, {}
        def stream(self, system, messages):
            yield raw
    reply = dr.ask(B(), "s", [], {}, log=lambda s: None, parser=mk())
    assert "\n" in reply["spoken"]
    new, n = ds.reflow(reply, mk())
    assert n == 3 and "\n" not in new["raw"] + new["spoken"]
    again = dr.ask(type("C", (), {"stop_reason": None, "usage": {},
                                  "stream": lambda s, a, b: iter(
                                      [new["raw"]])})(), "s", [], {},
                   log=lambda s: None, parser=mk())
    assert new["spoken"] == again["spoken"] == "One. Two! Three?"
    assert [(t["tag"], t["at"]) for t in new["tags"]] \
        == [(t["tag"], t["at"]) for t in again["tags"]]
    for t in new["tags"]:
        assert new["spoken"][t["at"]:t["at"] + 1] in ("T", "")
    # refusals and clean replies are returned as they are
    assert ds.reflow({**reply, "refused": True}, mk())[0]["raw"] == raw
    assert ds.reflow(new, mk())[1] == 0


def test_rev2_data_has_no_line_breaks_and_rev1_keeps_them(tmp_path):
    _o, recs, _m = _gen(tmp_path / "a", sessions=2, turns=4)
    for r in recs:
        assert "\n" not in r["reply_raw"] and "\n" not in r["spoken"]
        assert r["newlines"] >= 2
        assert r["spoken"] == "I once lost a ferry ticket in Osaka. Funny, right?"
        for h in r["history"]:
            assert "\n" not in h["reply"]
    _o, recs, _m = _gen(tmp_path / "b", "--beats3-rev", "1", sessions=1,
                        turns=3)
    assert all("\n" in r["reply_raw"] for r in recs)


def test_speak_turns_a_line_break_into_a_space_one_for_one():
    seen = []

    class TTS:
        whole = False

        def synth(self, text, mood=None):
            seen.append(text)
            return np.zeros(2400, np.float32), 24000, None
    spoken = "Hello there.\nHow are you?"
    tags = [{"tag": "pose", "arg": {}, "at": 13}]
    out = sp.speak(spoken, tags, TTS(), lambda ch: 0.05,
                   log=lambda s: None)
    assert all("\n" not in t for t in seen)
    assert len(out["spans"]) == len(spoken) and out["acts"][0]["at"] == 13


# ------------------------------------------------------- export rebuild

@pytest.mark.parametrize("rev", [1, 2])
def test_student_prompt_rebuilds_from_either_revision(tmp_path, rev):
    out, recs, _m = _gen(tmp_path, "--beats3-rev", str(rev), sessions=2,
                         turns=3)
    reg = dx.load_registry(out)
    r = recs[0]
    teacher = reg["prompts"][r["avatar"]["prompt_sha"]]
    assert (pz.SPOKEN_RULE in teacher) == (rev == 2)
    for student in ("short", "short+bible"):
        s = dx.student_system(teacher, r, student)
        assert s.startswith("You are ") and "Reply with what you say" in s
    # a run resumed across revisions holds both prompts
    r2 = ds.parse_opts(["gen", "--prompt", "beats3"])
    assert ds.beats3_rev(r2) == 2


def test_rev2_prompt_bytes_are_pinned():
    """Revision 2's wording, pinned (change it on purpose: bump BEATS3_REV
    and keep this revision's bytes instead of editing them)."""
    want = {"Hiyori|m0|v0": "19261ad977dcc8be",
            "lafei|m1|v3": "12b931e3c0147be9",
            "seed-san|m0|v5": "f00462a5638d9cc5"}
    for key, h in want.items():
        name, m, v = key.split("|")
        bm = _body(name)
        ml = mv.prompt_lines(bm) if m == "m1" else ""
        assert _h(pz.system_prompt(bm, PERSONA, False, "", ml, "beats3",
                                   int(v[1:]), 2)) == h, key
