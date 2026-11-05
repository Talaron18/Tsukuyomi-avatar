"""The rule graph and reminders (flow.rules, flow/rules.json, flow.live
--reminders): the detectors in en / ja / zh, the cap and the needs closure,
the rendered block, the injection with the memory notes, and off changing
nothing. No model, no GL."""

from __future__ import annotations

import json

import pytest

from flow import memory as mem
from flow import pose as pz
from flow import rules as rl
from tests.test_repair import live_performer, logs

G = rl.graph()


def ids(user, prev="", notes=None, cap=rl.CAP):
    return G.select(rl.Ctx(user, prev, notes), cap)


def notes(game=(), corrections=()):
    return mem.Notes(game=game, corrections=corrections)


# ------------------------------------------------------------ the graph

def test_the_shipped_graph_loads_and_covers_the_brief():
    order = [n["id"] for n in G.nodes]
    assert order == ["look-side", "her-right", "look-at-me", "wave-move",
                     "dance-frame", "place-glance", "correction",
                     "game-state", "quiet-chat", "language"]
    for n in G.nodes:
        assert {"id", "trigger", "text", "needs"} <= set(n) <= {
            "id", "trigger", "text", "needs", "requires"}
        assert n["trigger"] in rl.DETECTORS
        assert "\n" not in n["text"] and len(n["text"]) <= rl.TEXT_MAX
    assert G.byid["look-side"]["needs"] == ["her-right"]
    assert G.byid["her-right"]["trigger"] == "never"


def test_node_texts_use_the_beats2_wording():
    t = {n["id"]: n["text"] for n in G.nodes}
    look = pz.LOOK_RULE.replace("\n", " ")
    assert t["her-right"] in look
    assert "do it in one marker that holds that direction" in look
    assert 'with "hold" 2 to 4' in look and '"hold" 2 to 4' in t["look-side"]
    assert "The head stays turned until your next head move or the end of " \
           "the reply" in look
    assert "the head stays turned until your next head move or the end of " \
           "the reply" in t["look-side"]
    frame = pz.FRAME_ENTRY2.replace("\n", " ")
    assert "never for a wave or a single hand move" in frame
    assert "never [frame full] for a wave or a single hand move" in \
        t["wave-move"]
    assert "Write [frame full] right before" in frame
    assert "Write [frame full] right before a dance or another whole-body " \
           "move" in t["dance-frame"]
    glance = pz.GLANCE_RULE.replace("\n", " ")
    assert '(down for your feet, up for above you; your right is negative ' \
           '"head.yaw")' in glance and \
        '(down for your feet, up for above you; your right is negative ' \
        '"head.yaw")' in t["place-glance"]
    assert '"dur" 0.3' in glance and '"dur" 0.3' in t["place-glance"]
    quiet = pz.QUIET_RULE.replace("\n", " ")
    assert pz.QUIET_CHAT in t["quiet-chat"] and pz.QUIET_CHAT in quiet
    for phrase in ("means nobody has written for a while: keep talking as a "
                   "streamer does", "never say that the chat is quiet"):
        assert phrase in quiet and phrase in t["quiet-chat"]
    lang = "even if your earlier replies used another language"
    assert lang in pz.LANG_RULE2 and lang in t["language"]
    assert "NEXT MOVE" in mem.SYSTEM and '"next:"' in t["game-state"]
    assert "Work letters and numbers out yourself" in mem.SYSTEM


def test_a_bad_graph_is_refused():
    ok = {"id": "a", "trigger": "never", "text": "x", "needs": []}
    rl.Graph([ok])
    for bad, msg in [({**ok, "trigger": "nope"}, "no detector"),
                     ({**ok, "needs": ["b"]}, "no such node"),
                     ({**ok, "text": "two\nlines"}, "one short line"),
                     ({**ok, "text": "x" * 400}, "one short line"),
                     ({"id": "a"}, "no 'trigger'")]:
        with pytest.raises(ValueError, match=msg):
            rl.Graph([bad])
    with pytest.raises(ValueError, match="twice"):
        rl.Graph([ok, ok])


# ------------------------------------------------------------ detectors

def test_look_side_in_three_languages():
    for line in ("Look to your right!", "turn to your left please",
                 "右を見て", "左を向いて", "往右看", "向左转"):
        assert ids(line) == ["look-side", "her-right"], line
    for line in ("how are you", "my right hand hurts", "I like it",
                 "右手が痛い"):
        assert "look-side" not in ids(line), line


def test_look_at_me_in_three_languages():
    for line in ("Look at me!", "face the camera", "こっち見て",
                 "カメラ見て", "看着我", "看镜头"):
        assert ids(line) == ["look-at-me"], line
    for line in ("don't look at me", "look to your right", "hello"):
        assert "look-at-me" not in ids(line), line


def test_wave_and_dance_in_three_languages():
    for line in ("Can you wave at me?", "手を振って", "挥挥手"):
        assert ids(line) == ["wave-move"], line
    for line in ("Dance for me!", "ダンスして", "踊ってみせて", "跳个舞吧",
                 "来段舞"):
        assert ids(line) == ["dance-frame"], line
    for line in ("don't dance", "I went to dance class yesterday", "踊らない",
                 "别跳了"):
        assert "dance-frame" not in ids(line), line
    assert ids("wave and then dance") == ["wave-move", "dance-frame"]


def test_place_mention_in_three_languages():
    for line in ("I put the candy basket at your right foot",
                 "足元にお菓子を置いたよ", "你脚边有一个篮子",
                 "There's a cat behind you"):
        assert ids(line) == ["place-glance"], line
    assert ids("look behind you") != ["place-glance"]       # a request
    assert ids("my right hand hurts") == []


def test_correction_from_the_line_or_the_notes():
    assert ids("It's umamusume, not umasumume") == ["correction"]
    assert ids("不是苹果，是平果") == ["correction"]
    assert ids("ウマ娘だよ、うますむめじゃない") == ["correction"]
    n = notes(corrections=["umamusume, never umasumume"])
    assert ids("tell me about umasumume", notes=n) == ["correction"]
    assert ids("tell me about cats", notes=n) == []
    assert ids("hello") == []


def test_game_state_needs_the_notes():
    n = notes(game=["number game 1-20: 15 too big; next: her turn"])
    assert ids("is it 12?", notes=n) == ["game-state"]
    assert ids("is it 12?") == []
    assert ids("is it 12?", notes=notes()) == []


def test_quiet_chat_and_the_language_switch():
    assert ids(pz.QUIET_CHAT) == ["quiet-chat"]
    # a quiet turn never reads as a language switch or a request
    assert ids(pz.QUIET_CHAT, prev="こんにちは。") == ["quiet-chat"]
    prev_en = '[pose {"head.tilt": 4}] Hello there, nice to see you.'
    assert ids("こんにちは、元気？", prev=prev_en) == ["language"]
    assert ids("你好吗？", prev=prev_en) == ["language"]
    assert ids("Hello again!", prev=prev_en) == []         # no switch
    assert ids("Hello again!", prev="こんにちは。") == ["language"]
    assert ids("こんにちは") == []                         # no earlier reply


# ----------------------------------------------- the cap and the closure

def test_cap_is_three_with_the_needs_counted():
    line = "Look to your right, wave, and then dance"
    got = ids(line)
    assert got == ["look-side", "her-right", "wave-move"]    # dance: no room
    assert len(ids(line, cap=4)) == 4
    assert ids(line, cap=4)[-1] == "dance-frame"
    # five nodes fire, three are shown: the first three by priority
    many = ("Look at me and wave and dance! I put the basket behind you. "
            "No, it's umamusume, not umasumume")
    fired = G.fired(rl.Ctx(many))
    assert len(fired) >= 4 and len(G.select(rl.Ctx(many))) == 3
    assert G.select(rl.Ctx(many)) == fired[:3]


def test_a_node_comes_with_its_needs_or_not_at_all():
    nodes = [
        {"id": "a", "trigger": "asked_wave", "text": "A.", "needs": ["b", "c"]},
        {"id": "b", "trigger": "never", "text": "B.", "needs": ["d"]},
        {"id": "c", "trigger": "never", "text": "C.", "needs": []},
        {"id": "d", "trigger": "never", "text": "D.", "needs": []},
        {"id": "e", "trigger": "asked_wave", "text": "E.", "needs": []}]
    g = rl.Graph(nodes)
    ctx = rl.Ctx("wave")
    assert g.closure("a") == ["a", "b", "c", "d"]
    assert g.select(ctx, cap=4) == ["a", "b", "c", "d"]
    # the group does not fit in 3: skipped whole, the next node still shows
    assert g.select(ctx, cap=3) == ["e"]
    assert g.select(ctx, cap=0) == []


def test_deduplicated_and_in_priority_order():
    nodes = [
        {"id": "x", "trigger": "asked_wave", "text": "X.", "needs": ["s"]},
        {"id": "s", "trigger": "never", "text": "S.", "needs": []},
        {"id": "y", "trigger": "asked_wave", "text": "Y.", "needs": ["s"]}]
    g = rl.Graph(nodes)
    assert g.select(rl.Ctx("wave")) == ["x", "s", "y"]
    # a cycle in `needs` does not loop
    g2 = rl.Graph([{"id": "p", "trigger": "asked_wave", "text": "P.",
                    "needs": ["q"]},
                   {"id": "q", "trigger": "never", "text": "Q.",
                    "needs": ["p"]}])
    assert g2.select(rl.Ctx("wave")) == ["p", "q"]


# ------------------------------------------------------------ the block

def test_the_rendered_block():
    text, got = rl.block("Look to your right!")
    assert got == ["look-side", "her-right"]
    assert text == ("(reminders: " + G.byid["look-side"]["text"] + " "
                    + G.byid["her-right"]["text"] + ")")
    assert text.startswith("(reminders: ") and text.endswith(")")
    assert "\n" not in text
    assert rl.block("How are you?") == ("", [])
    assert rl.join("(stream notes: facts: viewer Ken)", text, "") == \
        "(stream notes: facts: viewer Ken)\n" + text
    assert rl.join("", "") == ""


# ----------------------------------------------- in the live loop

REPLY = 'Sure. [pose {"head.yaw": -15}] There. Anything else?'
NOTES = json.dumps({"game": ["number game, 1-20: 15 too big; next: her turn"],
                    "corrections": [], "facts": []})


class Cons:
    def stream(self, system, messages):
        yield NOTES


def memory():
    return mem.Memory("notes", 800, Cons(), timeout=5)


def last_user(st):
    return st.requests[-1][-1]["content"]


def test_reminders_ride_the_latest_user_message_with_the_notes(tmp_path):
    m = memory()
    p, hub, st, rec = live_performer(tmp_path, REPLY, memory=m,
                                     reminders="auto")
    p.turn("it is 12?")
    m.wait(10)
    p.turn("Look to your right")
    sent = last_user(st)
    lines = sent.split("\n")
    assert lines[0] == "Look to your right"
    note = [x for x in lines if x.startswith("(stream notes: ")]
    rem = [x for x in lines if x.startswith("(reminders: ")]
    assert len(note) == 1 and len(rem) == 1
    assert lines.index(note[0]) + 1 == lines.index(rem[0]) == len(lines) - 1
    assert "number game" in note[0]
    text, got = rl.block("Look to your right", REPLY, m.current())
    assert got == ["look-side", "her-right", "game-state"]
    assert rem[0] == text
    # the history keeps the plain message, and only the last one has it
    assert p.messages[-2]["content"].startswith("Look to your right")
    for req in st.requests:
        assert all("(reminders: " not in x["content"] for x in req[:-1])
    assert all("(reminders: " not in x["content"] for x in p.messages)
    assert any(x == "rules: turn 2: look-side, her-right, game-state"
               for x in logs(hub))
    print("\n--- tail block of the user message ---\n" + sent)


def test_game_reminder_comes_from_the_notes(tmp_path):
    m = memory()
    p, hub, st, rec = live_performer(tmp_path, REPLY, memory=m,
                                     reminders="auto")
    p.turn("let's play a number game")
    m.wait(10)
    p.turn("is it 12?")
    assert "(reminders: " + G.byid["game-state"]["text"] + ")" in \
        last_user(st)
    assert "rules: turn 2: game-state" in logs(hub)


def test_reminders_alone_and_none_when_nothing_fires(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, REPLY, reminders="auto")
    p.turn("Can you wave?")
    assert last_user(st).endswith(rl.block("Can you wave?")[0])
    p.turn("How are you?")
    assert "(reminders" not in last_user(st)
    assert not [x for x in logs(hub) if x == "rules: turn 2: "]


def test_the_budget_counts_the_reminders(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, REPLY, reminders="auto")
    p.turn("Look to your right")
    sent = len(last_user(st))
    # the request the backend got is what the budget estimated
    assert p.note_block and p.note_block in last_user(st)
    plain = len("Look to your right")
    assert sent > plain + len(p.note_block) - 1
    p.messages.append({"role": "user", "content": "x"})
    with_block = p.sent_chars()
    p.note_block = ""
    assert with_block - p.sent_chars() == len(rl.block(
        "Look to your right")[0]) + 1


def test_the_prefix_stays_byte_stable_with_reminders(tmp_path):
    p, hub, st, rec = live_performer(tmp_path, REPLY, reminders="auto",
                                     history_markers=0)
    p.history_turns = 0
    for line in ("Look to your right", "How are you?", "Can you wave?",
                 "Dance for me", "Thanks"):
        p.turn(line)
    for a, b in zip(st.requests, st.requests[1:]):
        # everything but the previous last user line is the same bytes
        assert b[:len(a) - 1] == a[:-1]
        assert a[-1]["content"].startswith(b[len(a) - 1]["content"])


def test_off_changes_nothing(tmp_path):
    base = live_performer(tmp_path / "a", REPLY)
    off = live_performer(tmp_path / "b", REPLY, reminders="off")
    assert base[0].reminders == "off"
    for p, hub, st, rec in (base, off):
        for line in ("Look to your right", "Can you wave?", pz.QUIET_CHAT):
            p.turn(line)
    assert base[2].requests == off[2].requests
    for _p, hub, st, _r in (base, off):
        assert all("(reminders" not in m["content"] for r in st.requests
                   for m in r)
        assert not [x for x in logs(hub) if x.startswith("rules:")]
    assert off[0].note_block == "" and off[0].rules_ids == []
    # with the notes alone the block is the notes' own bytes, as before
    m = memory()
    p, hub, st, rec = live_performer(tmp_path / "c", REPLY, memory=m)
    p.turn("it is 12?")
    m.wait(10)
    p.turn("Look to your right")
    assert last_user(st).split("\n")[-1] == m.block()
    assert "(reminders" not in last_user(st)


def test_reminders_need_body_mode(tmp_path):
    from tests.test_live import _performer
    p, hub, stream, clock, tts = _performer(tmp_path, "Hello there.",
                                            reminders="auto")
    p.turn("Look to your right")
    assert p.note_block == "" and p.rules_ids == []


def test_a_rule_the_body_cannot_follow_is_never_shown():
    # q8r3 A/B: "wave-move" told lafei (no wave on her menu) to write
    # [move wave], and the marker was dropped in 8 turns.
    from flow import body as fb
    from flow import rules as rl
    line = "can you wave at us and then dance?"
    lafei = rl.capabilities(fb.load("rigs/lafei.body.json"))
    hiyori = rl.capabilities(fb.load("rigs/Hiyori.body.json"))
    assert "move:wave" not in lafei and "move:wave" in hiyori
    assert "wave-move" not in rl.block(line, can=lafei)[1]
    assert "wave-move" in rl.block(line, can=hiyori)[1]
    assert "dance-frame" in rl.block(line, can=hiyori)[1]
    # a body without a framing switch gets no "[frame full]" rule
    assert "dance-frame" not in rl.block(line, can={"move:dance"})[1]
    # unknown body: everything allowed, as before
    assert {"wave-move", "dance-frame"} <= set(rl.block(line)[1])
