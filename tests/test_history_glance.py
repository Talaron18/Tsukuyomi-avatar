"""History markers, chunked trimming (a stable prefix), a trailing marker-only
segment, and the glance reflex: flow.live, flow.checks, flow.dpo, flow.pose."""

from __future__ import annotations

import argparse
import json

import pytest

from flow import checks as ck
from flow import director as dr
from flow import dpo
from flow import live as lv
from flow import memory as mem
from flow import pose as pz
from flow import stream as fs
from tests.test_live import FakeClock, SpeakingTTS, _events, _performer


class Student:
    stop_reason, usage = "end_turn", {}

    def __init__(self, reply="Okay. [nod] Right."):
        self.reply, self.requests = reply, []

    def stream(self, system, messages):
        self.requests.append([dict(m) for m in messages])
        yield self.reply


def plain(msgs):
    """Messages with the notes line cut off the last user message."""
    out = [dict(m) for m in msgs]
    c = out[-1]["content"]
    if "\n(stream notes" in c:
        out[-1]["content"] = c[:c.index("\n(stream notes")]
    return out


# ------------------------------------------------- A: history markers

def test_old_replies_are_sent_as_spoken_text_and_stored_unchanged(tmp_path):
    st = Student()
    p, hub, *_ = _performer(tmp_path, "", backend=st, history_markers=2,
                            trim_chunk=4)
    p.history_turns = 0
    for i in range(10):
        p.turn(f"line {i}")
    # the stored history keeps every marker
    assert all("[nod]" in m["content"] for m in p.messages
               if m["role"] == "assistant")
    for req in st.requests:
        a = [m["content"] for m in req if m["role"] == "assistant"]
        # the last two replies always have their markers
        assert all("[nod]" in x for x in a[-2:])
        # at most N + chunk - 1 = 5 keep them, the older ones are spoken text
        assert sum("[nod]" in x for x in a) <= 5
        assert all("[" not in x for x in a[:-5])
    last = [m["content"] for m in st.requests[-1] if m["role"] == "assistant"]
    assert last[0] == "Okay. Right."


def test_history_markers_zero_keeps_all(tmp_path):
    st = Student()
    p, *_ = _performer(tmp_path, "", backend=st, history_markers=0)
    p.history_turns = 0
    for i in range(8):
        p.turn(f"line {i}")
    assert all("[nod]" in m["content"] for m in st.requests[-1]
               if m["role"] == "assistant")


def test_the_budget_counts_what_is_sent(tmp_path):
    st = Student("Okay. " + "[nod]" * 40 + " Right.")
    p, *_ = _performer(tmp_path, "", backend=st, history_markers=1,
                       trim_chunk=1)
    p.history_turns = 0
    for i in range(4):
        p.turn(f"line {i}")
    stored = len(p.run["system"]) + sum(len(m["content"])
                                        for m in p.messages)
    assert p.sent_chars() < stored
    assert p.sent_chars() == len(p.run["system"]) + sum(
        len(m["content"]) for m in p.sent_view())


# --------------------------------------------- B: chunked trimming

def trims(hub):
    return [e["text"] for e in _events(hub, "log")
            if "history trimmed" in e["text"]]


def test_a_trim_drops_a_chunk_and_the_prefix_holds_between_trims(tmp_path):
    st = Student()
    p, hub, *_ = _performer(tmp_path, "", backend=st,
                            history_markers=0, trim_chunk=4)
    p.history_turns = 12
    for i in range(30):
        p.turn(f"line {i}")
    t = trims(hub)
    assert t and "a chunk of 4 exchanges" in t[0] and "last 9 exchanges" in t[0]
    assert len(t) <= 30 // 4
    stable = changed = 0
    for a, b in zip(st.requests, st.requests[1:]):
        if b[:len(a) - 1] == a[:-1]:
            stable += 1
        else:
            changed += 1
    assert changed == len(t) and stable >= 20
    assert max(len(r) for r in st.requests) <= 2 * 12 + 1
    # the old behaviour: one exchange per trim, the prefix moves every turn
    (tmp_path / "c").mkdir()
    st1 = Student()
    p1, hub1, *_ = _performer(tmp_path / "c", "", backend=st1,
                              history_markers=0, trim_chunk=1)
    p1.history_turns = 12
    for i in range(30):
        p1.turn(f"line {i}")
    assert len(trims(hub1)) == 30 - 13


def test_the_chunk_is_a_third_of_a_short_window_at_most(tmp_path):
    st = Student()
    p, hub, *_ = _performer(tmp_path, "", backend=st, history_markers=0,
                            trim_chunk=4)
    p.history_turns = 5
    for i in range(10):
        p.turn(f"line {i}")
    assert len(trims(hub)) == 4                  # one at a time, as before
    assert max(len(r) for r in st.requests) == 11


def test_the_context_budget_trims_a_chunk_too(tmp_path):
    st = Student("word " * 40)
    p, hub, *_ = _performer(tmp_path, "", backend=st, history_markers=0,
                            trim_chunk=4)
    p.history_turns = 0
    p._ctx = lv.REPLY_RESERVE + lv.CTX_MARGIN + 700
    for i in range(14):
        p.turn(f"line {i}")
    t = [x for x in trims(hub) if "context" in x]
    assert t and "a chunk of up to 4" in t[0]
    sizes = [len(r) for r in st.requests]
    drops = [a - b for a, b in zip(sizes, sizes[1:]) if b < a]
    assert drops and max(drops) >= 4             # several messages at once
    assert p.sent_chars() / (p.cpt or lv.GUESS_CPT) <= p.prompt_budget()


def test_prefix_is_byte_identical_between_trims_with_markers_and_notes(
        tmp_path):
    st = Student()
    cons_notes = json.dumps({"game": ["number game: 15 too big"],
                             "corrections": [], "facts": []})

    class Cons:
        def stream(self, s, msgs):
            yield cons_notes
    m = mem.Memory("notes", 800, Cons(), timeout=5)
    p, hub, *_ = _performer(tmp_path, "", backend=st, memory=m,
                            history_markers=2, trim_chunk=4)
    p.history_turns = 12
    for i in range(34):
        p.turn(f"15 is too big {i}")
        m.wait(10)
    assert any("stream notes" in r[-1]["content"] for r in st.requests)
    changed = 0
    for a, b in zip(st.requests, st.requests[1:]):
        # the notes ride only the last user message
        assert all("stream notes" not in x["content"] for x in b[:-1])
        if b[:len(a) - 1] == a[:-1]:
            # the sent messages minus the last exchange are identical, and
            # the last user line is a prefix of what it became in history:
            # the notes are appended after the text, so a prefix cache
            # holds through it
            assert b[len(a) - 1]["content"] == plain(a)[-1]["content"]
            assert a[-1]["content"].startswith(b[len(a) - 1]["content"])
        else:
            changed += 1
    # moves: the marker boundary every ~4 turns and a trim every ~4 turns
    # (together at the trim), not every turn
    assert 1 <= changed <= 34 // 4 + 1


# ------------------------------------- C: a trailing marker-only segment

def body_performer(tmp_path, reply, **kw):
    from tests.test_pose import _body_rig
    model, rig, bm = _body_rig(tmp_path / "bodyrig")
    er = fs.EventRig(str(model), str(rig), "default", None, None, 30.0,
                     str(tmp_path), log=lambda s: None)
    clock = FakeClock(5.0)
    hub = lv.Hub(clock)
    stream = fs.MotionStream(er, seed=0, autonomic=False, log=lambda s: None)
    a = argparse.Namespace(
        seed=0, lead=0.25, gap=0.12, lipsync="open", mouth_gain=1.0,
        cjk_rate=dr.CJK_RATE, latin_rate=dr.LATIN_RATE, mood="auto",
        head_carry=False, backend="deepseek", body_prompt="v1", frame="full",
        **kw)
    run = {"system": "sys", "mode": "body", "body": bm,
           "labels": dr.gesture_labels(["home.motion3.json"]),
           "offered": None,
           "parser": lambda: pz.parser_for(bm, gestures={}, moves=False)}
    st = Student(reply)
    p = lv.Performer(a, run, er, stream, clock, hub, st, SpeakingTTS(clock),
                     log=lambda s: None)
    return p, hub, st, stream


TRAIL = ('Oh nice! [pose {"head.yaw": 4}] Thank you. '
         '[pose {"head.yaw": -4, "hold": 0.3}]')


def logs(hub):
    return [e["text"] for e in _events(hub, "log")]


def test_a_trailing_pose_is_merged_not_planned_alone(tmp_path):
    p, hub, st, stream = body_performer(tmp_path, TRAIL)
    p.turn("hello")
    t = logs(hub)
    assert any("trailing markers merged" in x for x in t)
    assert not any(x.rstrip().endswith("plan 0.00s") or ": " == x[-2:]
                   for x in t if "cut" in x and "plan" in x)
    plans = [x for x in t if "live: turn 1." in x and " cut " in x]
    assert len(plans) == 2                          # two spoken sentences
    sent = [e for e in _events(hub, "audio")]
    assert len(sent) == 2
    # the pose is on the stream (applied at the end of the speech)
    ys = [k for k in stream.pending if k.get("type") == "pose"]
    assert len(ys) >= 2


def test_a_trailing_pose_is_dropped_when_a_release_follows(tmp_path):
    p, hub, st, stream = body_performer(tmp_path, TRAIL)
    p.carry = True
    p.turn("hello")
    t = logs(hub)
    assert any("trailing markers with no speech dropped" in x for x in t)
    assert not any("trailing markers merged" in x for x in t)
    assert len([e for e in _events(hub, "audio")]) == 2
    # the stored reply is untouched
    assert p.messages[-1]["content"] == TRAIL


def test_a_reply_of_markers_only_is_still_planned(tmp_path):
    p, hub, st, stream = body_performer(tmp_path, '[pose {"head.yaw": 4}]')
    p.turn("hello")
    assert any("live: turn 1.1" in x for x in logs(hub))


# ------------------------------------------------------- D: the glance

EN_YES = ["I put the candy basket at your right foot",
          "There's a cat on your left", "I hid a present behind you",
          "A balloon is above you", "I left the bag at your feet"]
JA_YES = ["足元にお菓子を置いたよ", "右足のところに置いたよ",
          "あなたの後ろにいるよ", "左側に置いた", "君の右側に置いたよ"]
ZH_YES = ["你脚边有一个篮子", "我把糖放在你右边", "身后有只猫",
          "你左边有只小猫", "你头顶有只鸟"]
NO = ["my right hand hurts", "turn right at the station", "it's on my right",
      "Look to your right!", "look at the basket at your feet",
      "右手が痛い", "私の右側に置いた", "足元を見て",
      "我右边有只猫", "往你右边看一下", "你的右手好漂亮", "右转",
      "your right hand is cute", "hello there", "こんにちは", "你好"]


@pytest.mark.parametrize("line", EN_YES + JA_YES + ZH_YES)
def test_places_are_found(line):
    assert ck.mentioned_place(line) is not None, line


@pytest.mark.parametrize("line", NO)
def test_other_lines_are_not_places(line):
    assert ck.mentioned_place(line) is None, line


def test_place_sides_and_targets():
    pl = ck.mentioned_place("I put the candy basket at your right foot")
    assert pl == {"side": -1, "vert": -1, "behind": False}
    t = ck.glance_targets(pl)
    assert t["head.yaw"] < 0 and t["gaze.x"] < 0                # her right
    assert t["head.pitch"] < 0 and t["gaze.y"] < 0              # down
    left = ck.glance_targets(ck.mentioned_place("a cat on your left"))
    assert left["head.yaw"] > 0 and "head.pitch" not in left
    up = ck.glance_targets(ck.mentioned_place("a balloon above you"))
    assert up["head.pitch"] > 0 and up["gaze.y"] > 0
    assert ck.glance_targets(ck.mentioned_place("behind you"))["head.yaw"] \
        < 0
    assert ck.mentioned_place("足元にお菓子")["vert"] == -1
    assert ck.mentioned_place("你左边有猫")["side"] == 1
    for pl in map(ck.mentioned_place, EN_YES + JA_YES + ZH_YES):
        assert ck.looks_toward(ck.glance_targets(pl), pl)


def test_looks_toward_needs_the_right_sign():
    pl = ck.mentioned_place("at your right foot")
    assert ck.looks_toward({"head.yaw": -10}, pl)
    assert not ck.looks_toward({"head.yaw": 10}, pl)
    assert ck.looks_toward({"gaze.y": -0.4}, pl)
    assert not ck.looks_toward({"head.yaw": -2, "gaze.y": 0.1}, pl)


GLANCE_REPLY = ('Oh, thank you! [pose {"head.tilt": 3}] That is so kind. '
                '[pose {"head.tilt": -3}] Really.')


def test_the_reply_opens_with_a_glance(tmp_path):
    p, hub, st, stream = body_performer(tmp_path, GLANCE_REPLY)
    p.turn("There's a cat on your left")
    t = logs(hub)
    g = [x for x in t if x.startswith("glance:")]
    assert len(g) == 1 and "her left" in g[0] and "head.yaw" in g[0]
    first = [x for x in t if "live: turn 1.1 pose" in x or
             "head.yaw" in x and "live: turn 1.1" in x]
    assert first
    p.turn("how are you")                       # no place: no glance
    assert len([x for x in logs(hub) if x.startswith("glance:")]) == 1


def test_no_glance_when_her_first_pose_already_looks_there(tmp_path):
    p, hub, st, stream = body_performer(
        tmp_path, '[pose {"head.yaw": 12}] Oh, thanks!')
    p.turn("There's a cat on your left")
    t = [x for x in logs(hub) if x.startswith("glance:")]
    assert len(t) == 1 and "skipped" in t[0]


def test_glance_off_and_explicit_look_and_quiet(tmp_path):
    p, hub, st, stream = body_performer(tmp_path, GLANCE_REPLY, glance="off")
    p.turn("There's a cat on your left")
    assert not [x for x in logs(hub) if x.startswith("glance:")]
    p, hub, st, stream = body_performer(tmp_path / "b", GLANCE_REPLY)
    p.turn("Look to your left, there's a cat")
    assert not [x for x in logs(hub) if x.startswith("glance:")]
    p.turn("(chat is quiet)", quiet=True)
    assert not [x for x in logs(hub) if x.startswith("glance:")]


def test_glance_carry_sends_the_head_home(tmp_path):
    p, hub, st, stream = body_performer(tmp_path, GLANCE_REPLY)
    p.carry = True
    p.turn("There's a cat on your left")
    g = [x for x in logs(hub) if x.startswith("glance:")]
    assert g and g[0].endswith("then home")


# ------------------------------------------------- E: training side

def rec(user, raw, quiet=False):
    return ck.make_rec(raw, user, "Mao", "s", 1, rid="r", quiet=quiet or None)


def test_checker_glance_at_mention():
    user = "I put the candy basket at your right foot"
    good = rec(user, 'Oh! [pose {"head.yaw": -9, "head.pitch": -10}] Thanks.')
    second = rec(user, 'Oh! [pose {"brow.L.lift": 0.5}] [pose {"gaze.y": '
                       '-0.4}] Thanks.')
    wrong = rec(user, 'Oh! [pose {"head.yaw": 9}] Thanks.')
    late = rec(user, 'Oh! [pose {"brow.L.lift": 0.5}] [pose {"head.tilt": 2'
                     '}] [pose {"head.yaw": -9}] Thanks.')
    nop = rec(user, "Oh, thanks!")
    na = rec("hello there", 'Hi! [pose {"head.yaw": 9}]')
    asked = rec("Look to your right at the basket", 'Hi! [pose {}]')
    quiet = rec(user, "Hm.", quiet=True)
    recs = []
    for i, r in enumerate([good, second, wrong, late, nop, na, asked,
                           quiet]):
        r["id"] = f"r{i}"
        recs.append(r)
    v, extra, why = ck.check_glance_at_mention(recs, None)
    assert [v[f"r{i}"] for i in range(8)] == [
        "pass", "pass", "fail", "fail", "fail", "na", "na", "na"]
    assert why["r4"] == "no_pose" and why["r2"] == "no_glance"
    assert "glance_at_mention" in ck.CHECKS
    assert ("glance_at_mention", None) in ck.rows_of({}) or any(
        k == "glance_at_mention" for k, _ in ck.rows_of({}))


def test_dpo_glance_pair_adds_the_beat_to_the_teacher_reply():
    raw = 'Oh, thanks! [pose {"head.tilt": 3}] So kind.'
    user = "I put the candy basket at your right foot"
    chosen, rejected = dpo.glance_at_mention(raw, user)
    assert rejected == raw and chosen != raw
    assert chosen.startswith('[pose {"head.yaw": -8')
    assert chosen.endswith(raw)
    assert dpo._rec(chosen, user)["spoken"] == dpo._rec(raw, user)["spoken"]
    v, _, _ = ck.check_glance_at_mention(
        [ck.make_rec(chosen, user, "x", "s", 1, rid="c"),
         ck.make_rec(rejected, user, "x", "s", 1, rid="r")], None)
    assert v == {"c": "pass", "r": "fail"}
    # only the variables the body has
    c2, _ = dpo.glance_at_mention(raw, user, offered=["head.yaw"])
    assert '"head.pitch"' not in c2 and '"head.yaw"' in c2
    assert dpo.glance_at_mention(raw, user, offered=["head.tilt"]) is None


def test_dpo_glance_pair_reverses_a_glance_the_teacher_made():
    raw = 'Oh! [pose {"head.yaw": -12, "gaze.x": -0.4}] Thanks!'
    user = "I put the candy at your right"
    chosen, rejected = dpo.glance_at_mention(raw, user)
    assert chosen == raw
    assert '"head.yaw": 12' in rejected and '"gaze.x": 0.4' in rejected
    assert dpo._rec(rejected, user)["spoken"] == \
        dpo._rec(raw, user)["spoken"]
    assert dpo.glance_at_mention(
        'Oh! [pose {"head.yaw": -30}] Thanks!', "something behind you") \
        is None
    assert dpo.glance_at_mention("no markers", user) is None
    assert dpo.glance_at_mention(raw, "hello there") is None
    assert dpo.glance_at_mention(raw, "Look to your right!") is None
    assert "glance_at_mention" in dpo.KINDS
    out = dpo.synthetic_pair("glance_at_mention", raw, user)
    assert out == (chosen, rejected)


def test_the_place_topic_is_beats2_only():
    path = f"{lv.os.path.dirname(lv.__file__)}/datagen.personas.json"
    cfg = json.load(open(path, encoding="utf-8"))
    (t,) = [x for x in cfg["topics"] if x["id"] == "place"]
    assert t["prompts"] == ["beats2"] and t["weight"] > 0
    assert t["examples"] and all(ck.mentioned_place(e) or True
                                 for e in t["examples"])
    # the example lines are places, none asks to look
    assert sum(ck.mentioned_place(e) is not None for e in t["examples"]) >= 5


def test_the_glance_rule_is_in_rev_3_only():
    from tests.test_pose import _body_rig  # noqa: F401 - the fixture rig
    from flow import body as fb
    import os
    bm = fb.load(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "rigs", "Hiyori.body.json"))
    r1 = pz.system_prompt(bm, "x", False, "", "", "beats2", 0, 1)
    r2 = pz.system_prompt(bm, "x", False, "", "", "beats2", 0, 2)
    r3 = pz.system_prompt(bm, "x", False, "", "", "beats2", 0, 3)
    assert pz.GLANCE_RULE not in r1 and pz.GLANCE_RULE not in r2
    assert pz.GLANCE_RULE in r3 and r3.replace(pz.GLANCE_RULE, "") == r2
    assert pz.BEATS2_REV == 3
    assert pz.system_prompt(bm, "x", False, "", "", "beats2", 0) == r3
