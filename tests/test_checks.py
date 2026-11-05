"""
flow.checks: the failure checkers, on small synthetic records (both signs
of a look, all three languages, a body without arms, the template repeated),
the loaders for an evalsft file, dataset shards and a live turns.jsonl, and
the table of several inputs. No network, no model.
"""

from __future__ import annotations

import itertools
import json
import os

import pytest

from flow import checks as ck
from flow import pose as pz
from tests.test_beats import FACES
from tests.test_beats2 import face_bm  # noqa: F401 - the fixture
from tests.test_dataexport import _run

BODIES = ck.Bodies(budgets={
    "arms": {"hand_moves": {"bust": 1, "full": 1}, "frames": ["bust", "full"]},
    "armless": {"hand_moves": {"bust": 0, "full": 0},
                "frames": ["bust", "full"]},
    "vrm": {"hand_moves": {"bust": 0, "full": 3}, "frames": ["full"]}})


def pose(**kw):
    kw = {k.replace("_", "."): v for k, v in kw.items()}
    dur = {k: kw.pop(k) for k in ("dur", "hold") if k in kw}
    return "[pose " + json.dumps({**kw, **dur}) + "]"


_TURN = itertools.count(1)


def rec(raw, user="Hello!", avatar="arms", session="s", turn=None, **kw):
    """A record; turns count up by default so every id is its own."""
    return ck.make_rec(raw, user, avatar, session,
                       next(_TURN) if turn is None else turn, **kw)


def verdict(name, *recs, bodies=BODIES):
    v, extra, why = ck.CHECKS[name](list(recs), bodies)
    return [v[r["id"]] for r in recs], extra, why


# -------------------------------------------------------------- look_sign

LOOKS = [
    ("Look to your right!", "R"), ("Could you turn to the left?", "L"),
    ("Glance to the right for me", "R"), ("look left", "L"),
    ("右を向いてみて", "R"), ("左を見て！", "L"),
    ("往右看一下", "R"), ("向左转", "L"), ("看你的右边", "R"),
    ("轉到左邊看看", "L"), ("看向右邊", "R"),
    # the character's own side without a verb (q8r3 turned to the viewer's)
    ("そっちの右側", "R"), ("君の左側を", "L"), ("あなたの右側！", "R"),
    ("ラフィーから見て右側", "R"), ("你的左边", "L"), ("你的右邊。", "R"),
    ("そっちの右側にいるよ", None), ("君の右手", None), ("你的右手", None),
    ("Look to your left, then to your right", None),   # both sides
    ("Point at your left elbow with your right hand", None),
    ("Look at your right hand", None),                   # a body part
    ("右手を上げて", None), ("You're right!", None), ("Hello", None)]


@pytest.mark.parametrize("line,side", LOOKS)
def test_asked_side(line, side):
    assert ck.asked_side(line) == side


def test_look_sign_both_signs_and_all_languages():
    # right is negative, left positive
    lines = {"Look to your right": -1, "Look to your left": 1,
             "右を向いて": -1, "左を見て": 1, "往右看": -1, "向左转": 1,
             "看右邊": -1, "轉到左邊": 1}
    for line, sign in lines.items():
        good = rec(pose(head_yaw=15 * sign, gaze_x=0.6 * sign) + " Sure.",
                   line)
        bad = rec(pose(head_yaw=-15 * sign) + " Sure.", line)
        gaze_bad = rec(pose(gaze_x=-0.5 * sign) + " Sure.", line)
        small = rec(pose(head_yaw=-3 * sign, brow_L_lift=0.5)
                    + pose(head_yaw=12 * sign) + " Sure.", line)
        got, _extra, _why = verdict("look_sign", good, bad, gaze_bad, small)
        # a yaw under 7 degrees is no turn: the first real one decides
        assert got == ["pass", "fail", "fail", "pass"], line


def test_look_sign_na_and_no_turn():
    got, extra, why = verdict(
        "look_sign",
        rec(pose(head_yaw=15) + " Hi.", "Nice weather"),
        rec(pose(head_yaw=15) + " Hi.", "Look left and look right"),
        rec(pose(head_tilt=6) + " Hi.", "Look to your left"),
        rec(pose(head_yaw=-15) + pose(head_yaw=15) + " Hi.",
            "Look to your right"))
    assert got == ["na", "na", "fail", "pass"]
    assert extra == {"no_turn": 1} and list(why.values()) == ["no_turn"]
    # the first marker decides, and a marker whose two disagree fails
    a = rec(pose(head_yaw=15, gaze_x=-0.6) + " Hi.", "Look to your left")
    assert verdict("look_sign", a)[0] == ["fail"]
    assert verdict("look_sign", a)[2][a["id"]] == "wrong_sign"


# ------------------------------------------------------------ the hands

def test_wave_obeyed_and_armless():
    wave = rec('[move {"name": "wave"}] Hi!', "Wave at me!")
    hand = rec(pose(hand_R_y=0.1) + " Hi!", "手を振って")
    none = rec(pose(head_tilt=5) + " Hi!", "挥手")
    other = rec('[move {"name": "sway"}] Hi!', "Please wave")
    armless = rec(pose(head_tilt=5) + " Hi!", "Wave!", avatar="armless")
    off = rec(pose(head_tilt=5) + " Hi!", "Nice day")
    got, _e, _w = verdict("wave_obeyed", wave, hand, none, other, armless,
                          off)
    assert got == ["pass", "pass", "fail", "fail", "na", "na"]
    for line in ("バイバイ！", "揮手一下", "wave", "waving"):
        assert ck.asked_wave(line)
    assert not ck.asked_wave("wavelength")


def test_body_request_obeyed():
    asked = "Can you raise your hand and show me?"
    hand = rec(pose(hand_L_y=0.1) + " Sure.", asked)
    move = rec('[move {"name": "clap"}] Sure.', asked)
    gest = rec("[gesture:wave_a] Sure.", asked)
    face = rec(pose(brow_L_lift=0.6) + " Sure.", asked)
    armless = rec(pose(brow_L_lift=0.6) + " Sure.", asked, avatar="armless")
    plain = rec(pose(brow_L_lift=0.6) + " Sure.", "How was your day?")
    nod = rec(pose(head_pitch=-10) + " Sure.", "Can you nod?")
    topic = rec(pose(brow_L_lift=0.6) + " Sure.", "Hmm?", topic="pose")
    limb = rec(pose(brow_L_lift=0.6) + " Sure.", "Show me your hands?",
               topic="show")
    quiet = rec("Hmm.", pz.QUIET_CHAT)
    got = verdict("body_request_obeyed", hand, move, gest, face, armless,
                  plain, nod, topic, limb, quiet)[0]
    # a topic alone asks for nothing; show / pose also need a limb word
    assert got == ["pass", "pass", "pass", "fail", "na", "na", "na", "na",
                   "fail", "na"]
    # ja and zh
    ja = rec(pose(brow_L_lift=0.6) + " うん。", "両手を上げて！")
    zh = rec(pose(hand_R_y=0.2) + " 好。", "举起手来")
    assert verdict("body_request_obeyed", ja, zh)[0] == ["fail", "pass"]


# --------------------------------------------------------------- openings

Q8R2 = pose(brow_L_lift=0.8, brow_R_lift=0.8, head_pitch=5, head_yaw=-5,
            dur=0.2, hold=0.5)


def test_opening_share_and_the_q8r2_template():
    rs = [rec(Q8R2 + " Hi.", turn=1), rec(Q8R2 + " Hi.", turn=2),
          rec(pose(brow_L_lift=0.5, brow_R_lift=0.5, head_pitch=5,
                   head_yaw=-5) + " Hi.", turn=3),         # brows too low
          rec(pose(brow_L_lift=0.8, brow_R_lift=0.8, head_pitch=5,
                   head_yaw=-5, head_tilt=3) + " Hi.", turn=4),  # extra var
          rec(pose(brow_L_lift=0.8, brow_R_lift=0.8, head_pitch=5,
                   head_yaw=4) + " Hi.", turn=5),          # yaw sign
          rec("Hi.", turn=6),
          rec(pose(mouth_form=0.5) + " Hi.", turn=7)]
    v, extra, _w = ck.check_opening(rs, BODIES)
    assert [v[r["id"]] for r in rs] == [
        "fail", "fail", "pass", "pass", "pass", "na", "pass"]
    # 6 openings with a pose: 4 share the brow+pitch+yaw set (3 +  1 of tilt)
    assert extra["top_set"]["share"] == pytest.approx(
        4 / 6) or extra["top_set"]["share"] == pytest.approx(3 / 6)
    assert extra["top_exact"]["share"] == pytest.approx(2 / 6)
    assert extra["top_exact"]["opening"]["head.yaw"] == -5
    agg = ck.aggregate(rs, BODIES, ["opening"])["opening"]
    assert agg["fail"] == 2 and agg["n"] == 6 and agg["na"] == 1


def test_template_repeat_within_a_session():
    a = pose(brow_L_lift=0.6) + " One. " + pose(mouth_form=0.5) + " Two."
    b = pose(brow_L_lift=0.1) + " Un. " + pose(mouth_form=0.9) + " Deux."
    c = pose(brow_L_lift=0.6) + " One. " + pose(head_tilt=4) + " Two."
    rs = [rec(a, session="s1", turn=1), rec(b, session="s1", turn=2),
          rec(c, session="s1", turn=3), rec(c, session="s1", turn=4),
          rec("No marker.", session="s1", turn=5),
          rec(c, session="s1", turn=6),
          rec(a, session="s2", turn=1), rec(a, session="s2", turn=2)]
    got = verdict("template_repeat", *rs)[0]
    # the turn after a markerless one is compared with the last marked one
    assert got == ["na", "fail", "pass", "fail", "na", "fail", "na", "fail"]
    # sessions do not mix, and the order is by turn, not by position
    swapped = [rs[1], rs[0]]
    assert verdict("template_repeat", *swapped)[0] == ["fail", "na"]


# -------------------------------------------------------------- languages

def test_language_en_ja_zh():
    cases = [
        ("How are you today?", "I am fine, thank you.", "pass"),
        ("How are you today?", "元気だよ、ありがとう。", "fail"),
        ("今日はいい天気ですね", "そうですね、散歩しましょう。", "pass"),
        ("今日はいい天気ですね", "是啊，今天天气真好。", "fail"),
        ("今天天气真好啊", "是啊，出去走走吧。", "pass"),
        ("今天天气真好啊", "Yes, let's go out.", "fail"),
        ("Tell me about 東京タワー and 渋谷 please", "Sure!", "na"),
        ("OK!", "Okay.", "na"),
        (pz.QUIET_CHAT, "Hello there!", "na")]
    rs = [rec(pose(head_tilt=4) + " " + sp, u, turn=i + 1, spoken=sp)
          for i, (u, sp, _w) in enumerate(cases)]
    assert verdict("language", *rs)[0] == [w for *_x, w in cases]
    assert ck.mixed("Tell me about 東京タワー and 渋谷 please")
    assert not ck.mixed("元気？ OK")


# ---------------------------------------------------------------- frames

def test_frame_for_dance_and_before_wave():
    dance = '[move {"name": "dance", "beats": 8}]'
    full = "[frame full]"
    rs = [rec(full + dance + " Yay!"),            # framed
          rec(pose(head_tilt=4) + dance + " Yay!"),     # in bust, no frame
          rec(dance + " Yay!", avatar="vrm"),     # never switches
          rec(dance + " Yay!", start="full"),     # starts in full
          rec('[move {"name": "sway"}] Yay!'),    # not a whole-body move
          rec(dance + full + " Yay!")]            # frame after the move
    assert verdict("frame_for_dance", *rs)[0] == [
        "pass", "fail", "na", "na", "na", "fail"]
    wave = '[move {"name": "wave"}]'
    ws = [rec(wave + " Hi!"), rec(full + wave + " Hi!"),
          rec(wave + full + " Hi!"), rec("Hi!")]
    assert verdict("frame_before_wave", *ws)[0] == [
        "pass", "fail", "pass", "na"]


# ----------------------------------------------------------------- beats

def test_beats_per_sentence_and_reply():
    two = (pose(brow_L_lift=0.6, head_tilt=3) + " One! "
           + pose(eye_L_smile=0.8, head_tilt=-3) + " Two!")
    one = pose(brow_L_lift=0.6) + " One! Two!"
    hand = pose(hand_R_y=0.1) + " One. Two."
    v, extra, _w = ck.check_beats([rec(two), rec(one), rec(hand),
                                   rec(pose(brow_L_lift=0.6))], BODIES)
    assert list(v.values()) == ["pass", "fail", "fail", "na"]
    assert extra["per_reply"] == pytest.approx((2 + 1 + 0) / 3)
    assert extra["per_sentence"] == pytest.approx(3 / 6)


# ---------------------------------------------------------------- quiet

def test_quiet_says_quiet_all_languages():
    q = pz.QUIET_CHAT
    cases = [("Hmm, the chat is quiet today.", "fail"),
             ("It seems no one is talking.", "fail"),
             ("Let me tell you about my morning.", "pass"),
             ("チャットが静かだね。", "fail"),
             ("そういえば昨日ね。", "pass"),
             ("聊天室好安静啊。", "fail"), ("没人说话呢。", "fail"),
             ("我来讲个故事吧。", "pass")]
    rs = [rec("x", q, turn=i + 1, spoken=sp)
          for i, (sp, _w) in enumerate(cases)]
    assert verdict("quiet_says_quiet", *rs)[0] == [w for _s, w in cases]
    assert verdict("quiet_says_quiet",
                   rec("x", "Hello", spoken="the chat is quiet"))[0] == ["na"]


# ------------------------------------------------- bodies, loaders, table

def test_bodies_read_the_budget_from_the_rig_files(face_bm):  # noqa: F811
    b = ck.Bodies(rigs=os.path.join(os.path.dirname(__file__), "..", "rigs"))
    assert b.has_hands("Hiyori") and b.can_frame("Hiyori")
    assert not b.can_frame("seed-san")
    assert b.has_hands("nobody") and b.can_frame("nobody")
    from flow import framing as fr
    # a budget measured from a body (flow.framing.budget) is read as such
    face = ck.Bodies(budgets={"face": fr.budget(face_bm)})
    assert face.has_hands("face") == bool(
        fr.budget(face_bm)["hand_moves"]["full"])


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_loaders_and_the_table(tmp_path, capsys):
    # an evalsft folder: student and recorded are two inputs
    ev = tmp_path / "eval_x"
    ev.mkdir()
    rows = [{"id": "Hiyori.a.0001/t01", "session": "Hiyori.a.0001",
             "turn": 1, "avatar": "Hiyori", "user": "Look to your right",
             "reply_raw": pose(head_yaw=-15) + " Sure.", "spoken": "Sure."},
            {"id": "Hiyori.a.0001/t02", "session": "Hiyori.a.0001",
             "turn": 2, "avatar": "Hiyori", "user": "Look to your left",
             "reply_raw": pose(head_yaw=-15) + " Sure.", "spoken": "Sure."}]
    _write(ev / "student.jsonl", rows)
    _write(ev / "recorded.jsonl", rows[:1])
    got = ck.load(str(ev))
    assert [lab for lab, _ in got] == ["eval_x:student", "eval_x:recorded"]
    assert [len(r) for _l, r in got] == [2, 1]
    assert ck.load(str(ev / "student.jsonl"))[0][0] == "eval_x/student"
    # a live turns.jsonl: raw reply, quiet, framing, no avatar
    live = tmp_path / "live"
    live.mkdir()
    _write(live / "turns.jsonl", [
        {"turn": 1, "user": "hi", "reply": Q8R2 + " Hi!", "framing": "bust"},
        {"turn": 2, "user": pz.QUIET_CHAT, "quiet": True,
         "reply": "The chat is quiet.", "framing": "bust"}])
    (lab, lr), = ck.load(str(live))
    assert lr[0]["avatar"] == "live" and lr[1]["quiet"]
    assert lr[0]["start"] == "bust" and lr[1]["session"] == lr[0]["session"]
    assert ck.load(str(live / "turns.jsonl"), "Mao")[0][1][0]["avatar"] \
        == "Mao"
    # the table: one column per input, per avatar blocks
    labels = ["a", "b"]
    res = [ck.run(got[0][1], BODIES, "avatar"),
           ck.run(lr, BODIES, "avatar")]
    text = ck.tables(labels, res, "avatar")
    assert "look_sign" in text and "50% (1/2)" in text
    assert "[Hiyori]" in text and "[live]" in text
    assert text.splitlines()[0].split() == labels
    # the CLI, and the json
    out = tmp_path / "o.json"
    assert ck.main([str(ev / "student.jsonl"), str(live), "--by", "avatar",
                    "--json", str(out), "--rigs", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "eval_x/student" in printed and "live" in printed
    d = json.loads(out.read_text())
    assert d["eval_x/student"]["checks"]["look_sign"]["fail"] == 1
    assert d["eval_x/student"]["by_avatar"]["Hiyori"]["n"] == 2
    assert d["live"]["checks"]["quiet_says_quiet"]["fail"] == 1
    with pytest.raises(SystemExit):
        ck.main([str(ev / "student.jsonl"), "--checks", "nope"])


def test_dataset_shards_are_an_input(tmp_path):
    out, recs = _run(tmp_path, FACES, avatars="seed-san,Hiyori", sessions=2,
                     turns=3, prompt="beats2")
    (lab, got), = ck.load(out)
    assert len(got) == len(recs) and {r["avatar"] for r in got} == {
        "seed-san", "Hiyori"}
    assert got[0]["start"] in ("bust", "full") and got[0]["topic"]
    shard = sorted(os.listdir(os.path.join(out, "shards")))[0]
    (_l, one), = ck.load(os.path.join(out, "shards", shard))
    assert len(one) <= len(got)
    res = ck.run(got, ck.Bodies(data=out), "avatar")
    # every check runs on mocked replies of faces only
    assert res["checks"]["beats"]["n"] > 0
    assert res["checks"]["template_repeat"]["fail"] >= 1
    assert set(res["by_avatar"]) == {"seed-san", "Hiyori"}


# The turns of the r3 run (beats2) the old checker got wrong, paraphrased:
# an invitation, a tease, small talk, a head request, a compliment, the past.
NOT_ASKED = [
    ("Let's go to the park together!", "invite"),
    ("Do you want to come to the festival with me?", "invite"),
    ("Your pointing in the last stream was so cute", "tease"),
    ("That peace sign photo is a keeper", "small_talk"),
    ("Look to your right and follow it with your eyes", "look"),
    ("Nod if you agree", None),
    ("指差してる姿がかわいいね", "tease"),
    ("你比耶的样子好可爱", "tease"),
    ("手を振ってる写真、まだ持ってるよ", "small_talk"),
    ("I waved at you yesterday, remember?", "small_talk"),
    ("How was your day?", None),
]
# The real skips: explicit requests the teacher answered with the head only.
ASKED = [
    "Hey, point to where the sun is!",
    "Could you point me toward the best stall?",
    "Can you point at the tree?",
    "Keep that hand up and point to the left",
    "Point at something behind you",
    "Make a peace sign for the camera!",
    "あっち指差して！", "ピースしてみて", "手を振ってくれる？",
    "给我比个耶", "挥挥手吧", "請舉起手來", "雙手叉腰",
    "Show me how big it was with your hands",
    "Give me a thumbs up", "cross your arms", "Please wave",
]


def test_body_request_counts_explicit_hand_requests_only():
    from flow import handask as ha
    for line, topic in NOT_ASKED:
        assert not ck.asked_hands(line, topic), line
        assert not ha.asked_hand_action(line, topic), line
    for line in ASKED:
        assert ck.asked_hands(line), line
        assert ck.asked_hands(line, "invite"), line
    assert not ck.asked_hands("Please wave", quiet=True)
    # head only
    for line in ("Look at me", "Turn around!", "点头", "こっち見て"):
        assert not ck.asked_hands(line, "show"), line
    # the numbers: the false positives are na, the skips fail
    recs = [rec(pose(head_yaw=-10) + " Sure.", line, topic=topic)
            for line, topic in NOT_ASKED]
    recs += [rec(pose(head_yaw=-10) + " Sure.", line) for line in ASKED]
    got = verdict("body_request_obeyed", *recs)[0]
    assert got == ["na"] * len(NOT_ASKED) + ["fail"] * len(ASKED)
    done = [rec(pose(hand_R_y=0.3) + " Sure.", line) for line in ASKED]
    assert set(verdict("body_request_obeyed", *done)[0]) == {"pass"}


def test_wave_obeyed_ignores_a_wave_that_is_no_request():
    for line in ("I waved at you yesterday", "the heat wave is awful",
                 "手を振ってる姿が好き", "你挥手的样子很可爱"):
        assert not ck.asked_wave(line), line


# ------------------------------------------------------------- look_at_me

AT_ME = [
    ("Look at me!", True), ("look at the camera", True), ("Face me.", True),
    ("Can you look at me for a sec?", True), ("hey, look at chat", True),
    ("Turn to face me please", True), ("look over here", True),
    ("こっち見て", True), ("こっちを見て！", True), ("私を見て", True),
    ("カメラ見てよ", True), ("カメラを見てください", True),
    ("看着我", True), ("看我", True), ("看镜头", True), ("看著鏡頭笑一下", True),
    ("Don't look at me", False), ("stop looking at me", False),
    ("why are you looking at me?", False), ("こっち見ないで", False),
    ("别看我", False), ("不要看镜头", False), ("看我的绝招", False),
    ("Look at your right hand", False), ("look at this!", False),
    ("Look to your left", False),           # look_sign's, not look_at_me's
    ("Look at me, then to your right", False),
    ("右を向いて", False), ("Hello", False), (pz.QUIET_CHAT, False)]


@pytest.mark.parametrize("line,want", AT_ME)
def test_asked_at_me(line, want):
    assert ck.asked_at_me(line) is want


def test_look_at_me_is_not_a_side_request():
    for line, want in AT_ME:
        if want:
            assert ck.asked_side(line) is None, line


def test_look_at_me_checks_where_the_reply_ends_up():
    user = "Look at me!"
    ok = rec("Hi! " + pose(head_yaw=1, gaze_x=0.05) + " there.", user)
    none = rec("Hi there. " + pose(brow_L_lift=0.5), user)
    small = rec("Hi! " + pose(head_yaw=-5, gaze_x=-0.2), user)
    yaw = rec("Hi! " + pose(head_yaw=18) + " there", user)
    gaze = rec("Hi! " + pose(gaze_x=-0.6), user)
    both = rec("Hi! " + pose(head_yaw=-15, gaze_x=-0.5), user)
    # a glance away that is brought back: it ends on the viewer
    back = rec("Hm " + pose(head_yaw=15) + " ok " + pose(head_yaw=0), user)
    gone = rec("Hm " + pose(head_yaw=0) + " ok " + pose(head_yaw=15), user)
    elsewhere = rec("Hi! " + pose(head_yaw=18), "Hello!")
    got, extra, why = verdict("look_at_me", ok, none, small, yaw, gaze, both,
                              back, gone, elsewhere)
    assert got == ["pass", "pass", "pass", "fail", "fail", "fail", "pass",
                   "fail", "na"]
    assert [why[r["id"]] for r in (yaw, gaze, both)] == ["yaw", "gaze", "both"]
    assert extra == {"yaw": 2, "gaze": 1, "both": 1}
    q = rec("x " + pose(head_yaw=18), pz.QUIET_CHAT, quiet=True)
    assert verdict("look_at_me", q)[0] == ["na"]


def test_look_at_me_is_in_the_table():
    r = ck.run([rec("Hi " + pose(head_yaw=18), "Look at me!")], BODIES)
    assert r["checks"]["look_at_me"]["fail"] == 1
    assert "look_at_me" in ck.tables(["a"], [r])


# ------------------------------------------------------ correction_obeyed

CORR = [
    ("It's \"Umamusume\", not \"Umasumume\"!", None, [("Umasumume", "Umamusume")]),
    ("not umasumume but umamusume", None, [("umasumume", "umamusume")]),
    ("You said umasumume, it's umamusume", None, [("umasumume", "umamusume")]),
    ("ウマ娘だ！うますむめじゃない", None, [("うますむめ", "ウマ娘")]),
    ("うますむめじゃなくてウマ娘だよ", None, [("うますむめ", "ウマ娘")]),
    ("不是苹果，是梨", None, [("苹果", "梨")]),
    ("是梨不是苹果", None, [("苹果", "梨")]),
    ("你说苹果不对", None, [("苹果", None)]),
    ("違う、うまむすめ", "うますむめ、うますむめ！", [("うますむめ", "うまむすめ")]),
    ("no, umamusume!", "I love umasumume", [("umasumume", "umamusume")]),
    ("umamusume", "umasumume!", []),            # no cue: not a correction
    ("Hello there", "umasumume", []), ("What is not true is a lie", None, [])]


@pytest.mark.parametrize("line,prev,want", CORR)
def test_corrections(line, prev, want):
    assert ck.corrections(line, prev) == want


def test_correction_obeyed_fails_when_the_wrong_form_comes_back():
    s = "c"
    first = rec("Umasumume is great!", "Tell me", session=s, turn=1)
    bad = rec("Yes, umasumume!", "No, it's Umamusume, not Umasumume!",
              session=s, turn=2)
    good = rec("Right, Umamusume!", "It's Umamusume, not umasumume",
               session=s, turn=3)
    ja = rec("うますむめ、うますむめ！", "ウマ娘だ！うますむめじゃない",
             session=s, turn=4)
    ja_ok = rec("ウマ娘だね、ごめん！", "ウマ娘だ！うますむめじゃない",
                session=s, turn=5)
    before = rec("うますむめ、好き！", "なにが好き？", session="pc", turn=1)
    prevcase = rec("うますむめ！", "違う、うまむすめ", session="pc", turn=2)
    # the Y form alone does not count as the wrong X
    zh = rec("是梨！", "不是苹果，是梨", session="z", turn=1)
    zh_bad = rec("苹果真好吃", "不是苹果，是梨", session="z", turn=2)
    other = rec("Hi", "Hello!", session="o", turn=1)
    got = verdict("correction_obeyed", first, bad, good, ja, ja_ok, before,
                  prevcase, zh, zh_bad, other)[0]
    assert got == ["na", "fail", "pass", "fail", "pass", "na", "fail",
                   "pass", "fail", "na"]


# -------------------------------------------------------------- pose_copy

def _seq(*vals):
    return " ".join(f"word{i} " + pose(brow_L_lift=v, head_pitch=5)
                    for i, v in enumerate(vals))


def test_pose_copy_matches_or_mostly_matches_a_recent_reply():
    s = "p"
    r1 = rec(_seq(0.8, 0.9, 0.3, 0.8), session=s, turn=1)
    same = rec(_seq(0.8, 0.9, 0.3, 0.8), session=s, turn=2)
    mostly = rec(_seq(0.8, 0.9, 0.3, 0.1), session=s, turn=3)   # 3 of 4
    half = rec(_seq(0.8, 0.9, 0.1, 0.2), session=s, turn=4)     # 2 of 4
    one = rec(_seq(0.8), session=s, turn=5)                     # < 2 poses
    got, _x, _w = verdict("pose_copy", r1, same, mostly, half, one)
    assert got == ["na", "fail", "fail", "pass", "na"]
    # only the last 3 replies count
    f = [rec(_seq(*v), session="f", turn=i) for i, v in enumerate(
        [(0.8, 0.9), (0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.8, 0.9)])]
    assert verdict("pose_copy", *f)[0][-1] == "pass"
    f[-1] = rec(_seq(0.5, 0.6), session="f", turn=4)
    assert verdict("pose_copy", *f)[0][-1] == "fail"
    assert ck.copies(((("a", 1),), (("b", 2),)), ((("a", 1),), (("b", 2),)))
    assert not ck.copies((("a", 1),), (("a", 1),))
    # values are rounded, dur / hold ignored
    a = rec("x " + pose(brow_L_lift=0.81, dur=0.3) + " y "
            + pose(head_yaw=-5.2), session="q", turn=1)
    b = rec("x " + pose(brow_L_lift=0.8, dur=0.9) + " y "
            + pose(head_yaw=-5.4), session="q", turn=2)
    assert verdict("pose_copy", a, b)[0] == ["na", "fail"]
    # sessions are apart
    c = rec(_seq(0.8, 0.9), session="other", turn=1)
    d = rec(_seq(0.8, 0.9), session="other2", turn=2)
    assert verdict("pose_copy", c, d)[0] == ["na", "na"]


def test_corrections_ignore_ordinary_chat_and_catch_the_live_ones():
    from flow import checks as c
    # lines from the r3 teacher data the fuzzy fallback used to read as
    # corrections (that/what, Hand's/hand, ...)
    for line, prev in [
            ("Haha, well it's nice to meet you, Mao. So what do you like to "
             "do for fun around here?", "And that is what I do."),
            ("Hand's up, but it's not high enough. Give me a peace sign "
             "beside your face and hold it.", "There, my hand is up. Now?"),
            ("Ha, wrong — it was three. Which means you owe me that rewatch. "
             "Though honestly I think the stakes were rigged from the start.",
             "I always guess seven, with stakes."),
            ("写真に撮りたいわけじゃないんですけど、見てみたくて…！えっと、"
             "それじゃあ次はピースサイン、お願いしてもいいですか？", "うん")]:
        assert c.corrections(line, prev) == [], line
    # the live session's corrections of うますむめ / umasumume
    assert c.corrections("ウマ娘だ！うますむめじゃない", "") == [
        ("うますむめ", "ウマ娘")]
    assert c.corrections("umamusume, but you said umasumume", "") == [
        ("umasumume", "umamusume")]
    assert c.corrections("違う、うまむすめ", "うますむめ、ほんとだ！") == [
        ("うますむめ", "うまむすめ")]
