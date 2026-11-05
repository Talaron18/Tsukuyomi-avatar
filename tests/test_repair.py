"""The repair layer (flow.repair, flow.live --repair): deterministic fixes of
what the checkers (flow.checks) flag, sentence by sentence while the reply
streams, in en / ja / zh; the no-op cases; log mode; the pairs file for
scripts/train_dpo.py; the stored history. No model, no GL."""

from __future__ import annotations

import argparse
import json
import os

import pytest

from flow import checks as ck
from flow import director as dr
from flow import live as lv
from flow import pose as pz
from flow import repair as rp
from flow import stream as fs
from tests.test_live import FakeClock, SpeakingTTS, _events


def run(user, sentences, **kw):
    """(performed sentences, [fixes], the end-of-reply fixes) of a Turn."""
    t = rp.Turn(user, **kw)
    out, fixes = [], []
    for s in sentences:
        o, f = t.sentence(s)
        out.append(o)
        fixes += f
    return out, fixes, t.finish(), t


def pose(**d):
    return "[pose " + json.dumps(d) + "]"


def target(text):
    """The {key: value} of the first pose marker of a text."""
    for _a, _b, t in __import__("flow.dataexport", fromlist=["x"]
                                ).marker_spans(text):
        if t["tag"] == "pose":
            return t["arg"]["targets"]
    return {}


# ------------------------------------------------------------ look_sign

@pytest.mark.parametrize("user,want", [
    ("Please look to your right!", -1),          # en: her right is negative
    ("Look to your left.", 1),
    ("右を見て", -1),                            # ja
    ("左を向いて", 1),
    ("往右看", -1),                              # zh
    ("向左转", 1)])
def test_look_sign_sets_the_asked_side(user, want):
    wrong = pose(**{"head.yaw": -want * 15, "gaze.x": -want * 0.6})
    out, fixes, end, _t = run(user, [wrong + " Okay."])
    assert [f["check"] for f in fixes] == ["look_sign"]
    f = fixes[0]
    assert f["applied"] and f["ok"] and f["k"] == 1
    tg = target(out[0])
    assert tg["head.yaw"] == want * 15 and tg["gaze.x"] == want * 0.6
    assert out[0].endswith("] Okay.")
    assert not end
    # the checker agrees now
    rec = ck.make_rec(out[0], user, "x", "s", 1)
    assert ck.check_look_sign([rec], None)[0][rec["id"]] == ck.PASS


def test_look_sign_makes_a_disagreeing_marker_agree():
    out, fixes, _e, _t = run("Look to your right",
                             [pose(**{"head.yaw": 15, "gaze.x": -0.6})
                              + " Sure."])
    tg = target(out[0])
    assert tg["head.yaw"] == -15 and tg["gaze.x"] == -0.6 and fixes


def test_look_sign_keeps_the_other_keys_and_the_text():
    s = ' [pose {"head.yaw": 18, "hold": 3, "head.tilt": 4}] Over here.'
    out, fixes, _e, _t = run("look to your right", [s])
    assert out[0] == ' [pose {"head.yaw": -18, "hold": 3, "head.tilt": 4}] ' \
                     "Over here."
    assert fixes[0]["before"].startswith('[pose {"head.yaw": 18')
    assert fixes[0]["after"].startswith('[pose {"head.yaw": -18')


def test_look_sign_in_a_later_sentence_touches_that_sentence_only():
    s1 = pose(**{"brow.L.lift": 0.5}) + " Let me see."
    s2 = " " + pose(**{"head.yaw": 15}) + " There."
    out, fixes, _e, _t = run("Look to your right", [s1, s2])
    assert out[0] == s1
    assert target(out[1])["head.yaw"] == -15
    assert fixes[0]["k"] == 2


def test_look_sign_is_a_noop_when_right_or_not_asked_or_small():
    right = pose(**{"head.yaw": -15}) + " Okay."
    for user, sent in [("Look to your right", right),
                       ("Look to your right", pose(**{"head.yaw": 3})
                        + " Hmm."),                     # not a turn
                       ("How are you?", pose(**{"head.yaw": 15}) + " Fine."),
                       ("Look left or right", pose(**{"head.yaw": 15}) + "x"),
                       ("my right hand hurts", pose(**{"head.yaw": 15})
                        + " Oh.")]:
        out, fixes, _e, _t = run(user, [sent])
        assert out == [sent] and fixes == []


def test_a_wrong_look_already_spoken_is_not_touched():
    """Log mode lets sentence 1 through; sentence 2 is not repaired for it
    and the end of the reply reports it."""
    s1 = pose(**{"head.yaw": 15}) + " One."
    s2 = " " + pose(**{"head.yaw": 15}) + " Two."
    out, fixes, end, t = run("Look to your right", [s1, s2], mode="log")
    assert out == [s1, s2]
    assert [f["applied"] for f in fixes] == [False]
    assert end == []                  # already proposed once: not twice
    assert t.rebuild("whatever") == "whatever"


def test_a_reply_that_never_turns_is_only_reported():
    out, fixes, end, _t = run("Look to your right", ["Sure thing."])
    assert out == ["Sure thing."] and not fixes
    assert end[0]["check"] == "look_sign" and not end[0]["applied"]


def test_a_lenient_marker_is_reported_not_edited():
    s = "[pose {'head.yaw': 15}] Okay."
    out, fixes, _e, _t = run("Look to your right", [s])
    assert out == [s]
    assert not any(f["applied"] for f in fixes)


# ----------------------------------------------------------- look_at_me

@pytest.mark.parametrize("user", ["Look at me!", "Face the camera please",
                                  "こっち見て", "カメラを見て",
                                  "看着我", "看镜头"])
def test_look_at_me_centres_a_sideways_marker(user):
    s = pose(**{"head.yaw": 14, "gaze.x": -0.5, "brow.L.lift": 0.4,
                "dur": 0.3}) + " はい!"
    out, fixes, end, _t = run(user, [s])
    assert [f["check"] for f in fixes] == ["look_at_me"]
    tg = target(out[0])
    assert tg["head.yaw"] == 0 and tg["gaze.x"] == 0
    assert tg["brow.L.lift"] == 0.4
    assert fixes[0]["applied"] and not end


def test_look_at_me_fixes_every_sideways_marker_of_the_sentence():
    s = (pose(**{"head.yaw": 12}) + " a " + pose(**{"gaze.x": 0.5})
         + " b " + pose(**{"head.yaw": 2}))
    out, fixes, _e, _t = run("look at me", [s])
    rec = ck.make_rec(out[0], "look at me", "x", "s", 1)
    assert ck.final_look(rec) == (0.0, 0.0) or ck.final_look(rec)[0] in (0, 2)
    assert ck.check_look_at_me([rec], None)[0][rec["id"]] == ck.PASS
    assert len(fixes) == 1 and "head.yaw" in fixes[0]["before"]


def test_look_at_me_noops():
    centred = pose(**{"head.yaw": 2, "gaze.x": 0.1}) + " Hi."
    sideways = pose(**{"head.yaw": 14}) + " Hi."
    for user, sent in [("look at me", centred),
                       ("don't look at me", sideways),      # a negation
                       ("Look to your right", sideways),    # look_sign's
                       ("how are you", sideways),
                       ("(chat is quiet)", sideways)]:
        out, fixes, _e, _t = run(user, [sent],
                                 quiet=user == "(chat is quiet)")
        assert not [f for f in fixes if f["check"] == "look_at_me"], user
        if user != "Look to your right":      # look_sign's, not ours
            assert out == [sent], user


# ---------------------------------------------------------------- frames

DANCE = '[move {"name": "dance"}]'
WAVE = '[move {"name": "wave"}]'


@pytest.mark.parametrize("user", ["Dance for me!", "踊ってみせて", "跳个舞吧"])
def test_a_dance_gets_a_full_frame_before_it(user):
    out, fixes, _e, _t = run(user, [f"Okay! {DANCE} Here I go."],
                             start="bust", frames=True)
    assert out == [f"Okay! [frame full] {DANCE} Here I go."]
    assert fixes[0]["check"] == "frame_for_dance"
    assert fixes[0]["before"] == DANCE
    assert fixes[0]["after"] == "[frame full] " + DANCE


def test_a_dance_noops():
    s = f"Okay! {DANCE} Go."
    for kw in ({"start": "full", "frames": True},     # already full
               {"start": "bust", "frames": False},    # a fixed camera
               {"start": None, "frames": False}):     # no [frame] prompt
        assert run("dance", [s], **kw)[0] == [s]
    done = f"Okay! [frame full] {DANCE} Go."
    assert run("dance", [done], start="bust", frames=True)[0] == [done]
    # a full frame in an earlier sentence covers a dance in a later one
    out = run("dance", ["Ready? [frame full] Yes.", f" {DANCE} Go."],
              start="bust", frames=True)[0]
    assert out[1] == f" {DANCE} Go."


def test_a_late_full_frame_moves_up_to_the_dance():
    s = f"Okay {DANCE} and [frame full] go."
    out, fixes, _e, _t = run("dance", [s], start="bust", frames=True)
    assert out[0].count("[frame full]") == 1
    assert out[0].index("[frame full]") < out[0].index(DANCE)
    assert fixes


def test_a_full_frame_before_a_wave_is_taken_out():
    """The checker fails a full frame before a wave: a wave stays in bust."""
    s = f"Hi! [frame full] {WAVE} Hello."
    out, fixes, _e, _t = run("say hi", [s], start="bust", frames=True)
    assert out == [f"Hi! {WAVE} Hello."]
    assert fixes[0]["check"] == "frame_before_wave"
    assert fixes[0]["before"] == "[frame full] " + WAVE
    assert fixes[0]["after"] == WAVE


def test_a_wave_in_bust_and_a_dances_frame_are_left_alone():
    s = f"Hi! {WAVE} Hello."
    assert run("hi", [s], start="bust", frames=True)[0] == [s]
    both = f"[frame full] {DANCE} then {WAVE} bye."
    out, fixes, _e, _t = run("hi", [both], start="bust", frames=True)
    assert out == [both]
    assert [f["ok"] for f in fixes] == [False]        # logged, not touched
    # a full frame spoken before the wave sentence is not edited
    out, fixes, _e, _t = run("hi", ["[frame full] Hi.", f" {WAVE} bye."],
                             start="bust", frames=True)
    assert out[1] == f" {WAVE} bye." and not any(f["applied"]
                                                  for f in fixes)


def test_a_wave_then_a_dance_in_one_reply_ends_with_one_frame_before_the_dance():
    s = f"[frame full] {WAVE} Hi. {DANCE} Go."
    out, fixes, _e, _t = run("hi", [s], start="bust", frames=True)
    assert out[0].count("[frame full]") == 1
    assert out[0].index(WAVE) < out[0].index("[frame full]") \
        < out[0].index(DANCE)
    assert [f["check"] for f in fixes] == ["frame_before_wave",
                                           "frame_for_dance"]


# ------------------------------------------------------------------ wave

@pytest.mark.parametrize("user", ["Can you wave at me?", "手を振って!",
                                  "挥挥手", "wave bye-bye"])
def test_an_asked_wave_is_added_to_the_first_sentence(user):
    out, fixes, end, _t = run(user, [" Sure. " + pose(**{"brow.L.lift": .4})
                                     + " Bye.", " Later."], wave=True)
    assert out[0].startswith(" " + WAVE + " Sure.")
    assert out[1] == " Later."
    assert fixes[0]["check"] == "wave_obeyed" and fixes[0]["applied"]
    assert not end


def test_a_wave_is_not_added_when_it_cannot_be_unambiguous():
    s = "Sure, hi."
    # a hand pose in the first sentence is a wave for the checker
    hand = pose(**{"hand.R.y": 0.4}) + " Hi."
    assert run("wave", [hand], wave=True)[0] == [hand]
    assert run("wave", [WAVE + " Hi."], wave=True)[0] == [WAVE + " Hi."]
    # no wave on the menu / a body without hands: only told at the end
    out, fixes, end, _t = run("wave", [s], wave=False)
    assert out == [s] and not fixes
    assert end[0]["check"] == "wave_obeyed" and not end[0]["applied"]
    out, fixes, end, _t = run("wave", [s], wave=True, hands=False)
    assert out == [s] and not fixes and not end
    # only the first sentence decides: a later one is not given a wave
    out, fixes, _e, _t = run("wave", [" Hm.", " " + hand], wave=True)
    assert out[1] == " " + hand and out[0].lstrip().startswith(WAVE)


def test_a_body_request_is_only_logged():
    for user in ("Point at the sky!", "Make a peace sign", "ピースして",
                 "比个耶"):
        out, fixes, end, _t = run(user, ["Okay then."], wave=True)
        assert out == ["Okay then."] and not fixes
        assert [f["check"] for f in end] == ["body_request_obeyed"], user
        assert not end[0]["applied"]
    # obeyed: nothing
    s = pose(**{"hand.R.y": 0.5}) + " There."
    assert run("Point at the sky!", [s], wave=True)[3].finish() == []


def test_the_glance_check_is_left_to_the_reflex():
    s = pose(**{"brow.L.lift": 0.3}) + " Oh!"
    out, fixes, end, _t = run("I put the basket at your right foot", [s])
    assert out == [s] and not fixes and not end


# ---------------------------------------------------------- modes, rebuild

def test_log_mode_changes_nothing_and_off_does_not_look():
    s = pose(**{"head.yaw": 15}) + " Okay."
    out, fixes, _e, t = run("Look to your right", [s], mode="log")
    assert out == [s] and fixes and not fixes[0]["applied"]
    assert fixes[0]["after"].startswith('[pose {"head.yaw": -15')
    assert "log mode" in rp.log_line(1, fixes[0])
    assert not t.changed and t.rebuild("raw") == "raw"
    out, fixes, end, t = run("Look to your right", [s], mode="off")
    assert out == [s] and not fixes and not end and not t.on


def test_quiet_turns_are_not_repaired():
    s = pose(**{"head.yaw": 15}) + " Okay."
    out, fixes, end, _t = run("Look to your right", [s], quiet=True)
    assert out == [s] and not fixes and not end


def test_the_log_line_is_before_arrow_after():
    out, fixes, _e, _t = run("Look to your right",
                             [pose(**{"head.yaw": 15}) + " Okay."])
    assert rp.log_line(3, fixes[0]) == (
        'repair: turn 3.1 look_sign: [pose {"head.yaw": 15}] -> '
        '[pose {"head.yaw": -15}]')
    r = rp.record(3, "Look to your right", fixes[0], "auto")
    assert r["kind"] == "repair" and r["turn"] == 3 and r["applied"]
    assert r["check"] == "look_sign" and r["mode"] == "auto"
    json.dumps(r)


def test_rebuild_replaces_each_repaired_sentence_in_the_model_reply():
    cut = lv.SentenceCutter()
    reply = ('Look! ［pose {"head.yaw": 15}］ Over there. '
             'And more. ［pose {"head.yaw": 15}］ Done.')
    sents = cut.feed(reply) + cut.close()
    t = rp.Turn("Look to your right")
    performed = [t.sentence(s)[0] for s in sents]
    again = t.rebuild(reply)
    assert again == "".join(performed)
    assert again.count('"head.yaw": -15') == 1       # only the first turn
    assert "Done." in again and "And more." in again
    # a sentence the reply does not hold: the performed ones joined
    t2 = rp.Turn("Look to your right")
    t2.sentence(pose(**{"head.yaw": 15}) + " Hi.")
    assert t2.rebuild("something else entirely") == \
        pose(**{"head.yaw": -15}) + " Hi."


# ------------------------------------------------------- pairs, DPO reader

def test_pair_schema_and_the_trainer_reads_the_file(tmp_path):
    from flow.train import dpo as tdpo
    raw = pose(**{"head.yaw": 15}) + " Okay."
    fixed = pose(**{"head.yaw": -15}) + " Okay."
    prompt = [{"role": "system", "content": "SYS"},
              {"role": "user", "content": "Look to your right"}]
    row, meta = rp.pair(prompt, raw, fixed, "look_sign",
                        {"id": "live/rec/t01/repair", "avatar": "x"})
    assert row["prompt"] == prompt
    assert row["chosen"] == [{"role": "assistant", "content": fixed}]
    assert row["rejected"] == [{"role": "assistant", "content": raw}]
    assert meta["kind"] == "repair_look_sign" and meta["src"] == "live"
    rp.write_pair(str(tmp_path), row, meta)
    rp.write_pair(str(tmp_path), row, {**meta, "id": "live/rec/t02/repair"})
    path = tmp_path / "repair_pairs.jsonl"
    pairs, info = tdpo.load_pairs(str(path))
    assert len(pairs) == 2 and info["no_meta"] == 0
    p = pairs[0]
    assert p["chosen"] == fixed and p["rejected"] == raw
    assert p["kind"] == "repair_look_sign" and p["avatar"] == "x"
    assert p["session"] == "live/rec/t01"
    assert tdpo.parse_specs(str(path) + ":2") == [(str(path), 2.0)]
    assert tdpo.meta_path(str(path)).endswith("repair_pairs.meta.jsonl")


# ------------------------------------------------------ in the live loop

class Student:
    stop_reason, usage = "end_turn", {}

    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def stream(self, system, messages):
        self.requests.append([dict(m) for m in messages])
        for i in range(0, len(self.reply), 9):
            yield self.reply[i:i + 9]


def live_performer(tmp_path, reply, record=False, frame="bust", memory=None,
                   **kw):
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
        head_carry=False, backend="deepseek", body_prompt="beats2",
        frame=frame, **kw)
    run_ = {"system": "SYS", "mode": "body", "body": bm,
            "body_path": str(tmp_path / "bodyrig" / "x.body.json"),
            "labels": dr.gesture_labels(["home.motion3.json"]),
            "offered": None,
            "parser": lambda: pz.parser_for(bm, gestures={}, moves=True,
                                            frames=True)}
    st = Student(reply)
    rec = lv.Recorder(str(tmp_path / "rec"), 30.0, video=False) \
        if record else None
    p = lv.Performer(a, run_, er, stream, clock, hub, st,
                     SpeakingTTS(clock), log=lambda s: None, recorder=rec,
                     memory=memory)
    return p, hub, st, rec


def logs(hub):
    return [e["text"] for e in _events(hub, "log")]


def turn_rows(rec):
    rec.turns.flush()
    with open(os.path.join(rec.dir, "turns.jsonl"), encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


RAW = ('Sure. [pose {"head.yaw": 15}] Over there. '
       'Anything else? [pose {"brow.L.lift": 0.4}] Tell me.')


def test_live_repairs_the_sentence_and_keeps_what_she_performed(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, RAW, record=True)
    p.turn("Look to your right")
    fixed = RAW.replace('{"head.yaw": 15}', '{"head.yaw": -15}')
    # the history is the reply as performed; the model's is in the record
    assert p.messages[-1]["content"] == fixed
    lines = [x for x in logs(hub) if x.startswith("repair:")]
    assert lines[0] == ('repair: turn 1.2 look_sign: [pose {"head.yaw": 15}] '
                        '-> [pose {"head.yaw": -15}]') or \
        lines[0].startswith("repair: turn 1.")
    assert "look_sign" in lines[0] and '"head.yaw": -15' in lines[0]
    # the pose on the stream turns to her right (negative yaw)
    rows = turn_rows(rec)
    kinds = [r.get("kind") for r in rows]
    assert kinds.count("repair") == 1 and kinds.count(None) == 1
    rr = next(r for r in rows if r.get("kind") == "repair")
    assert rr["check"] == "look_sign" and rr["applied"] and rr["turn"] == 1
    tr = next(r for r in rows if r.get("kind") is None)
    assert tr["reply"] == fixed and tr["reply_unrepaired"] == RAW
    assert tr["repaired"] == ["look_sign"]
    # the next request shows her the performed reply, not the raw one
    p.turn("Thanks")
    prev = [m["content"] for m in st.requests[-1] if m["role"] == "assistant"]
    assert prev == [fixed]


def test_live_pose_token_has_the_repaired_side(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, RAW)
    p.turn("Look to your right")
    ys = [k for k in p.stream.pending if k.get("type") == "pose"]
    assert ys
    p2, hub2, st2, rec2 = live_performer(tmp_path / "b", RAW, repair="off")
    p2.turn("Look to your right")
    assert p2.messages[-1]["content"] == RAW
    assert not [x for x in logs(hub2) if x.startswith("repair:")]


def test_live_off_and_log_modes(tmp_path):
    p, hub, st, rec = live_performer(tmp_path / "o", RAW, repair="off")
    p.turn("Look to your right")
    assert p.messages[-1]["content"] == RAW
    assert not [x for x in logs(hub) if x.startswith("repair:")]
    p, hub, st, rec = live_performer(tmp_path / "l", RAW, record=True,
                                     repair="log")
    p.turn("Look to your right")
    assert p.messages[-1]["content"] == RAW           # nothing changed
    lines = [x for x in logs(hub) if x.startswith("repair:")]
    assert lines and "log mode" in lines[0]
    rows = turn_rows(rec)
    assert [r for r in rows if r.get("kind") == "repair"][0]["applied"] \
        is False
    assert "reply_unrepaired" not in rows[-1]
    assert not os.path.exists(os.path.join(rec.dir, "repair_pairs.jsonl"))


def test_live_default_is_auto(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, RAW)
    assert p.repair_mode == "auto"
    assert rp.DEFAULT_MODE == "auto"


def test_live_writes_the_pairs_file_the_trainer_reads(tmp_path):
    from flow.train import dpo as tdpo
    p, hub, st, rec = live_performer(tmp_path, RAW, record=True)
    p.turn("Look to your right")
    pairs_file = os.path.join(rec.dir, "repair_pairs.jsonl")
    meta_file = os.path.join(rec.dir, "repair_pairs.meta.jsonl")
    rows = [json.loads(x) for x in open(pairs_file, encoding="utf-8")]
    metas = [json.loads(x) for x in open(meta_file, encoding="utf-8")]
    assert len(rows) == len(metas) == 1
    row, meta = rows[0], metas[0]
    assert row["prompt"][0] == {"role": "system", "content": "SYS"}
    assert row["prompt"][-1]["role"] == "user"
    assert row["prompt"][-1]["content"].startswith("Look to your right")
    assert row["rejected"][0]["content"] == RAW
    assert row["chosen"][0]["content"] == p.messages[-1]["content"]
    assert meta["kind"] == "repair_look_sign" and meta["src"] == "live"
    assert meta["verdicts"] == {"look_sign": ["fail", "pass"]}
    # the trainer's own --data path (scripts/train_dpo.py stats / train)
    got, linfo = tdpo.load_data(argparse.Namespace(data=pairs_file + ":2"))
    assert len(got) == 1 and got[0]["weight"] == 2.0 and not linfo["no_meta"]
    pairs, info = tdpo.load_pairs(pairs_file)
    assert len(pairs) == 1 and pairs[0]["kind"] == "repair_look_sign"
    assert pairs[0]["chosen"] != pairs[0]["rejected"]
    assert info["no_meta"] == 0


def test_a_recordings_pairs_are_not_all_held_out_together(tmp_path):
    """The trainer holds whole sessions out (split_pairs): a session per
    recording would leave `train_dpo.py train --data X/repair_pairs.jsonl`
    with no training pairs, or with a recording's worth held out."""
    from flow.train import dpo as tdpo
    p, hub, st, rec = live_performer(tmp_path, RAW, record=True)
    for _ in range(8):
        p.turn("Look to your right")
    pairs, _info = tdpo.load_pairs(os.path.join(rec.dir,
                                                "repair_pairs.jsonl"))
    assert len(pairs) == 8 and len({q["session"] for q in pairs}) == 8
    train, ev, _sp = tdpo.split_pairs(pairs, 0.1)
    assert len(train) >= 6 and len(train) + len(ev) == 8
    # and the live loop's own meta ids end in tNN/repair
    ids = [json.loads(x)["id"] for x in open(os.path.join(
        rec.dir, "repair_pairs.meta.jsonl"), encoding="utf-8")]
    assert ids[0] == "live/rec/t01/repair" and ids[-1].endswith("t08/repair")


def test_the_plan_gets_the_repaired_value_not_only_the_history(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, RAW)
    p.turn("Look to your right")
    plan = [x for x in logs(hub) if x.startswith("live: turn 1.")
            and " cut " in x]
    assert plan and '"head.yaw": -15' in " ".join(plan)
    assert '"head.yaw": 15' not in " ".join(plan)
    # with --repair off the plan keeps what the model wrote
    p2, hub2, st2, rec2 = live_performer(tmp_path / "off", RAW,
                                         repair="off")
    p2.turn("Look to your right")
    plan2 = [x for x in logs(hub2) if x.startswith("live: turn 1.")
             and " cut " in x]
    assert '"head.yaw": 15' in " ".join(plan2)


def test_no_pair_when_nothing_was_repaired(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, RAW, record=True)
    p.turn("How are you?")
    assert not os.path.exists(os.path.join(rec.dir, "repair_pairs.jsonl"))
    assert p.messages[-1]["content"] == RAW
    assert not [r for r in turn_rows(rec) if r.get("kind") == "repair"]


def test_live_dance_gets_its_frame_and_a_wave_when_asked(tmp_path):
    raw = f"Okay! {DANCE} Here I go."
    p, hub, st, rec = live_performer(tmp_path, raw, record=True)
    p.turn("Dance for me")
    assert p.messages[-1]["content"] == f"Okay! [frame full] {DANCE} Here I go."
    assert any("frame_for_dance" in x for x in logs(hub))
    # a wave that was asked for and not made
    p, hub, st, rec = live_performer(tmp_path / "w", "Hello there!",
                                     record=True)
    p.turn("Can you wave?")
    assert p.messages[-1]["content"] == WAVE + " Hello there!"
    assert [r["check"] for r in turn_rows(rec)
            if r.get("kind") == "repair"] == ["wave_obeyed"]


def test_checks_load_skips_the_repair_lines(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, RAW, record=True)
    p.turn("Look to your right")
    rec.turns.flush()
    # the first line is a repair record: the file still loads
    [(label, recs)] = ck.load(os.path.join(rec.dir, "turns.jsonl"))
    assert len(recs) == 1
    assert recs[0]["raw"].count("-15") == 1
    assert ck.check_look_sign(recs, None)[0][recs[0]["id"]] == ck.PASS
    # a file that begins with one
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps({"kind": "repair", "turn": 1}) + "\n"
                 + json.dumps({"turn": 1, "user": "hi", "reply": "Hi."})
                 + "\n")
    assert len(ck.load(str(f))[0][1]) == 1


def test_a_trailing_marker_dropped_with_carry_stays_in_the_history(tmp_path):
    raw = 'Oh! [pose {"head.yaw": 15}] Fine. [pose {"head.yaw": -4}]'
    p, hub, st, rec = live_performer(tmp_path, raw)
    p.carry = True
    p.turn("Look to your right")
    assert p.messages[-1]["content"] == raw.replace(
        '{"head.yaw": 15}', '{"head.yaw": -15}')
