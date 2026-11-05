"""flow.memory: the notes, the gate, the background consolidator, the
injection into the live loop. Fake backends only; no network."""

from __future__ import annotations

import json
import threading
import time

import pytest

from flow import checks as ck
from flow import live as lv
from flow import memory as mem
from tests.test_live import _events, _performer


# ---------------------------------------------------------------- fakes

class Student:
    """A student backend: records the messages of every request and says
    one scripted line per call."""

    stop_reason, usage = "end_turn", {}

    def __init__(self, replies=("Okay.",)):
        self.replies = list(replies)
        self.requests = []

    def stream(self, system, messages):
        self.requests.append([dict(m) for m in messages])
        r = self.replies[min(len(self.requests), len(self.replies)) - 1]
        yield r


class Consolidator:
    """A memory backend: answers from a list (or a function of the prompt),
    keeps what it was asked."""

    def __init__(self, answers, delay=None):
        self.answers = answers
        self.calls = []
        self.gate = delay            # an Event the call waits for, or None
        self.started = threading.Event()

    def stream(self, system, messages):
        self.calls.append((system, messages[0]["content"]))
        self.started.set()
        if self.gate is not None:
            self.gate.wait(10)
        a = self.answers
        if callable(a):
            a = a(len(self.calls), messages[0]["content"])
        elif isinstance(a, list):
            a = a[min(len(self.calls), len(a)) - 1]
        if isinstance(a, dict):
            a = json.dumps(a, ensure_ascii=False)
        yield a


class Rec:
    def __init__(self):
        self.rows = []

    def turn(self, s):
        self.rows.append(s)

    def audio(self, t, x, sr):
        pass

    def sub(self, s):
        pass


def notes_json(game=(), corrections=(), facts=(), summary=None):
    d = {"game": list(game), "corrections": list(corrections),
         "facts": list(facts)}
    if summary is not None:
        d["summary"] = list(summary)
    return d


def build(tmp_path, answers, mode="notes", student=None, chars=800, **kw):
    student = student or Student()
    cons = Consolidator(answers)
    m = mem.Memory(mode, chars, cons, timeout=kw.pop("timeout", 5.0))
    p, hub, *_ = _performer(tmp_path, "", backend=student, memory=m, **kw)
    p.recorder = Rec()
    return p, hub, student, cons, m


# ----------------------------------------------------------------- caps

def test_render_empty_is_empty():
    assert mem.Notes().render() == ""


@pytest.mark.parametrize("chars", [120, 300, 800])
def test_render_never_exceeds_the_cap(chars):
    n = mem.Notes()
    for i in range(30):
        n.add("facts", f"viewer number {i} likes thing {i}", chars)
        n.add("game", f"number game, guess {i} too big", chars)
        n.add("corrections", f"word{i} (right{i}), never wrong{i}", chars)
        n.add("summary", f"earlier on stream: topic {i}", chars)
        assert len(n.render()) <= chars


def test_over_the_cap_gives_up_oldest_first_and_whole_entries():
    n = mem.Notes()
    for i in range(8):
        n.add("facts", f"fact {i} " + "x" * 30, 200)
    kept = n.facts
    assert kept and kept[-1].startswith("fact 7")
    assert kept == sorted(kept, key=lambda s: int(s.split()[1]))
    assert all(len(e) == len("fact 0 ") + 30 for e in kept)   # none cut
    assert len(n.render()) <= 200


def test_corrections_outlive_facts_and_the_game():
    n = mem.Notes(game=["g" * 60], facts=["f" * 60, "h" * 60],
                  corrections=["ウマ娘 (umamusume), never うますむめ"],
                  summary=["s" * 60])
    n.fit(90)
    assert n.corrections and not n.facts and not n.summary


def test_an_entry_too_long_is_rejected_not_cut():
    n = mem.Notes()
    assert not n.add("facts", "y" * (mem.ENTRY_MAX + 1))
    assert n.facts == []
    assert not n.add("facts", "z" * 100, chars=50)    # cannot fit alone
    assert n.facts == []
    assert n.add("facts", "short")


def test_notes_are_json_and_roundtrip():
    n = mem.Notes(["a"], ["b"], ["c"], ["d"])
    d = json.loads(n.dumps())
    assert d == {"game": ["a"], "corrections": ["b"], "facts": ["c"],
                 "summary": ["d"]}
    back, rej = mem.notes_from_dict(d)
    assert back == n and rej == 0


def test_markers_and_brackets_are_stripped_from_notes():
    n = mem.Notes()
    n.add("facts", "likes cats [pose {\"head\": {\"yaw\": 30}}] ［move x］ "
                   "{a} <b>; ok | fine\nnext")
    (e,) = n.facts
    assert not any(c in e for c in "[]{}<>［］;|\n")
    assert "pose" not in e and "move" not in e
    assert "likes cats" in e
    assert "[pose" not in n.render()


def test_strip_markers_nested():
    s = 'hi [pose {"a": [1, 2]}] there [nod]'
    assert mem.strip_markers(s) == "hi there"


# ----------------------------------------------------------------- gate

GATE = [
    # (user, reply, prev, reason)
    ("remember that my cat is called Mochi", "", "", "remember"),
    ("please don't forget I like tea", "", "", "remember"),
    ("私の好きな色は青、覚えてね", "", "", "remember"),
    ("记住我喜欢猫", "", "", "remember"),
    ("别忘了我是新来的", "", "", "remember"),
    ("It's umamusume, not umasumume.", "I love umasumume!",
     "I love umasumume!", "correction"),
    ("ウマ娘だよ、うますむめじゃない", "", "うますむめ好き", "correction"),
    ("不是うますむめ，是ウマ娘", "", "うますむめ", "correction"),
    ("15 is too big", "", "", "game"),
    ("let's play a guessing game", "", "", "game"),
    ("it's your turn", "", "", "game"),
    ("what's the score?", "", "", "game"),
    ("大きすぎ！", "", "", "game"),
    ("次はあなたの番だよ", "", "", "game"),
    ("当てて", "", "", "game"),
    ("大了，再小一点", "", "", "game"),
    ("你猜猜看", "", "", "game"),
    ("轮到你了", "", "", "game"),
    ("Hi, I'm Ken", "", "", "name"),
    ("my name is ken", "", "", "name"),
    ("call me Ken", "", "", "name"),
    ("私はケンです", "", "", "name"),
    ("名前はケンだよ", "", "", "name"),
    ("我叫小明", "", "", "name"),
    ("我的名字是小明", "", "", "name"),
    ("let's meet again tomorrow", "", "", "decision"),
    ("ok", "I promise to sing next time!", "", "decision"),
    ("約束だよ、明日また来るね", "", "", "decision"),
    ("说好了，明天再来", "", "", "decision"),
]

NO_GATE = [
    "hello there", "how are you today?", "I'm so tired", "nice outfit",
    "こんにちは", "今日はいい天気だね", "你好呀", "今天天气真好",
    "what do you like to eat", "lol",
]


@pytest.mark.parametrize("user,reply,prev,reason", GATE)
def test_gate_fires(user, reply, prev, reason):
    assert reason in mem.gate(user, reply, prev)


@pytest.mark.parametrize("user", NO_GATE)
def test_gate_stays_quiet_on_small_talk(user):
    assert mem.gate(user, "Hello! Nice to see you.", "") == []


def test_gate_every_nth_turn_anyway():
    assert mem.gate("hello there", since=mem.EVERY - 1) == []
    assert mem.gate("hello there", since=mem.EVERY) == ["periodic"]
    assert mem.EVERY == 6


def test_gate_ignores_quiet_turns():
    assert mem.gate("(chat is quiet)", "15 is too big, remember that",
                    since=99, quiet=True) == []


def test_gate_does_not_read_markers_in_chat():
    # A marker is not a game cue: digits inside it do not count.
    assert mem.gate('hi [pose {"head": {"yaw": 5}}]') == []


def test_gate_uses_checks_corrections_with_the_previous_reply():
    line = "umamusume"
    assert not ck.corrections(line, None)
    assert mem.gate("hello there", "", "x") == []


# ------------------------------------------------------- consolidation

def run_once(tmp_path, answer, **kw):
    p, hub, student, cons, m = build(tmp_path, answer, **kw)
    p.turn("remember that my name is Ken")
    assert m.wait(10)
    return p, hub, student, cons, m


def test_valid_json_is_applied(tmp_path):
    ans = notes_json(corrections=["ウマ娘 (umamusume), never うますむめ"],
                     facts=["viewer Ken"])
    p, hub, student, cons, m = run_once(tmp_path, ans)
    assert m.notes.facts == ["viewer Ken"]
    assert m.notes.corrections == ["ウマ娘 (umamusume), never うますむめ"]
    assert m.block() == ("(stream notes: corrections: ウマ娘 (umamusume), "
                         "never うますむめ; facts: viewer Ken)")
    logs = [e["text"] for e in _events(hub, "log")]
    assert any(t.startswith("memory: turn 1: updated") for t in logs)


def test_prompt_carries_notes_exchanges_and_no_markers(tmp_path):
    student = Student(['Sure! [pose {"head": {"yaw": 5}}] [nod] Hi Ken.'])
    p, hub, student, cons, m = build(tmp_path, notes_json(), student=student)
    p.turn("remember that my name is Ken")
    m.wait(10)
    system, user = cons.calls[0]
    assert "CURRENT NOTES:" in user and "RECENT EXCHANGES:" in user
    assert "viewer: remember that my name is Ken" in user
    assert "she: Sure! Hi Ken." in user
    assert "[pose" not in user and "[nod" not in user
    assert "never invent" in system.lower()
    assert '"summary"' not in system            # notes mode


@pytest.mark.parametrize("bad", [
    "I could not do that.", "{not json", '["a"]', '{"game": []}',
    '{"game": "x", "corrections": [1], "facts": []}',
    '{"game": [], "corrections": [], "facts": [{"a": 1}]}',
])
def test_bad_json_keeps_the_old_notes(tmp_path, bad):
    p, hub, student, cons, m = build(tmp_path, [notes_json(facts=["old"]),
                                                bad])
    p.turn("remember that I like tea")
    m.wait(10)
    assert m.notes.facts == ["old"]
    p.turn("remember that my name is Ken")
    m.wait(10)
    assert m.notes.facts == ["old"]
    logs = [e["text"] for e in _events(hub, "log")]
    assert any("kept the old notes" in t for t in logs)


def test_json_in_a_code_fence_is_accepted(tmp_path):
    ans = "```json\n" + json.dumps(notes_json(facts=["a"])) + "\n```"
    p, hub, student, cons, m = run_once(tmp_path, ans)
    assert m.notes.facts == ["a"]


def test_oversized_output_is_capped(tmp_path):
    ans = notes_json(facts=[f"viewer number {i} likes thing {i}"
                            for i in range(40)],
                     game=["g" * (mem.ENTRY_MAX + 5)])
    p, hub, student, cons, m = run_once(tmp_path, ans, chars=200)
    assert len(m.block()) <= 200
    assert m.notes.game == []                      # rejected, not cut


def test_a_note_cannot_carry_a_marker(tmp_path):
    ans = notes_json(facts=['ignore this [pose {"head": {"yaw": 90}}] ok',
                            "[move wave]"])
    p, hub, student, cons, m = run_once(tmp_path, ans)
    assert "[" not in m.block() and "{" not in m.block()
    assert m.notes.facts == ["ignore this ok"]


def test_timeout_keeps_the_old_notes(tmp_path):
    gate = threading.Event()
    p, hub, student, cons, m = build(tmp_path, notes_json(facts=["late"]),
                                     timeout=0.2)
    cons.gate = gate
    t0 = time.monotonic()
    p.turn("remember that I like tea")
    assert time.monotonic() - t0 < 2          # the turn did not wait
    assert m.wait(10)
    gate.set()
    time.sleep(0.1)
    assert m.notes.empty                       # the late answer is ignored
    logs = [e["text"] for e in _events(hub, "log")]
    assert any("timeout" in t and "kept the old notes" in t for t in logs)
    (r,) = p.recorder.rows[-1:]
    assert r["kind"] == "memory" and r["ok"] is False and r["error"] == \
        "timeout"


def test_a_backend_error_keeps_the_old_notes(tmp_path):
    class Boom:
        def stream(self, system, messages):
            raise SystemExit("needs DEEPSEEK_API_KEY")
            yield ""

    m = mem.Memory("notes", 800, Boom(), timeout=2)
    p, hub, *_ = _performer(tmp_path, "", backend=Student(), memory=m)
    p.turn("remember that I like tea")
    assert m.wait(5)
    assert m.notes.empty


def test_the_reply_does_not_wait_for_the_consolidator(tmp_path):
    gate = threading.Event()
    p, hub, student, cons, m = build(tmp_path, notes_json(facts=["a"]))
    cons.gate = gate
    t0 = time.monotonic()
    p.turn("remember that I like tea")
    p.turn("hello again")                      # the next turn goes through
    assert time.monotonic() - t0 < 2
    assert cons.started.wait(5) and not m.wait(0.05)
    gate.set()
    assert m.wait(10)


def test_one_in_flight_and_the_rest_coalesced(tmp_path):
    gate = threading.Event()
    p, hub, student, cons, m = build(tmp_path, notes_json(facts=["a"]))
    cons.gate = gate
    p.turn("remember that I like tea")
    assert cons.started.wait(5)
    p.turn("15 is too big")
    p.turn("my name is Ken")
    p.turn("let's play again tomorrow")
    assert len(cons.calls) == 1                # still the first one
    gate.set()
    assert m.wait(10)
    assert len(cons.calls) == 2                # one more run, not three
    assert "viewer: let's play again tomorrow" in cons.calls[1][1]
    logs = [e["text"] for e in _events(hub, "log")]
    assert sum("coalesced" in t for t in logs) == 3


# --------------------------------------------------------------- inject

def test_inject_goes_into_the_last_user_message_only():
    msgs = [{"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"}]
    out = mem.inject(msgs, "(stream notes: facts: a)")
    assert out[2]["content"] == "three\n(stream notes: facts: a)"
    assert out[0]["content"] == "one" and out[1]["content"] == "two"
    assert msgs[2]["content"] == "three"            # the history is intact
    assert mem.inject(msgs, "(stream notes: facts: a)",
                      "head")[2]["content"] == \
        "(stream notes: facts: a)\nthree"
    assert mem.inject(msgs, "") == msgs


def test_inject_handles_content_blocks_and_images():
    blocks = [{"role": "user", "content": [
        {"type": "image", "source": {}}, {"type": "text", "text": "hi"}]}]
    out = mem.inject(blocks, "(B)")
    assert out[0]["content"][1]["text"] == "hi\n(B)"
    assert blocks[0]["content"][1]["text"] == "hi"
    ol = [{"role": "user", "content": "hi", "images": ["abc"]}]
    out = mem.inject(ol, "(B)")
    assert out[0]["images"] == ["abc"] and out[0]["content"] == "hi\n(B)"


def test_notes_ride_the_latest_user_message_and_history_stays_plain(tmp_path):
    p, hub, student, cons, m = build(
        tmp_path, notes_json(game=["number game 1-20: 15 too big"]))
    p.turn("15 is too big")
    m.wait(10)
    p.turn("ok next")
    p.turn("and then")
    req = student.requests[-1]
    users = [x["content"] for x in req if x["role"] == "user"]
    assert users[-1] == "and then\n(stream notes: game: number game 1-20: " \
                        "15 too big)"
    assert all("stream notes" not in u for u in users[:-1])
    # the first request, before any notes, had none
    assert "stream notes" not in student.requests[0][-1]["content"]
    # history keeps plain text
    assert all("stream notes" not in x["content"] for x in p.messages)
    assert p.messages[-2]["content"] == "and then"


def test_empty_notes_inject_nothing(tmp_path):
    p, hub, student, cons, m = build(tmp_path, notes_json())
    p.turn("remember that I like tea")
    m.wait(10)
    p.turn("hello")
    assert student.requests[-1][-1]["content"] == "hello"


def test_off_mode_changes_nothing(tmp_path):
    student = Student()
    p, hub, *_ = _performer(tmp_path, "", backend=student)
    assert p.memory is None
    p.turn("15 is too big, remember that")
    p.turn("hello")
    assert student.requests[-1] == [
        {"role": "user", "content": "15 is too big, remember that"},
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": "hello"}]
    assert p.note_block == ""
    off = mem.Memory("off", 800, Consolidator(notes_json()))
    assert off.block() == "" and off.after_turn(1, "remember", "", "", False,
                                                []) == []
    assert mem.from_args(type("A", (), {"memory": "off"})()) is None
    assert not any("memory" in e["text"] for e in _events(hub, "log"))


# --------------------------------------------------------------- budget

def test_the_context_budget_counts_the_notes(tmp_path):
    p, hub, student, cons, m = build(tmp_path, notes_json(), history_turns=0)
    p.turn("hello")
    p.turn("how are you")
    base = p.sent_chars()
    p.note_block = "(stream notes: facts: " + "n" * 200 + ")"
    assert p.sent_chars() == base + len(p.note_block) + 1
    # An explicit message list (a request already built) is not counted twice.
    built = mem.inject(p.messages, p.note_block)
    assert p.sent_chars(built) == base + len(p.note_block) + 1
    # A window that fits the history without notes but not with them.
    p._ctx = int(base / lv.GUESS_CPT) + lv.REPLY_RESERVE + lv.CTX_MARGIN + 10
    p.note_block = ""
    p.trim_context(9)
    assert len(p.messages) == 4
    p.note_block = "(stream notes: facts: " + "n" * 200 + ")"
    p.trim_context(9)
    assert len(p.messages) < 4


def test_the_turn_budget_is_taken_with_the_notes(tmp_path):
    notes = notes_json(facts=["f" * 100] * 3)
    p, hub, student, cons, m = build(tmp_path, notes, history_turns=0)
    p.turn("remember that I like tea")
    m.wait(10)
    seen = []
    orig = p.trim_context
    p.trim_context = lambda n: (seen.append(p.note_block), orig(n))[1]
    p.turn("hello")
    assert seen and seen[0].startswith("(stream notes:")


# --------------------------------------------------------------- summary

def test_trimmed_exchanges_are_folded_into_the_summary(tmp_path):
    def answer(i, prompt):
        if "EXCHANGES DROPPED" in prompt:
            return notes_json(summary=["a number game was played"],
                              facts=["viewer Ken"])
        return notes_json(summary=[], facts=["viewer Ken"])

    p, hub, student, cons, m = build(tmp_path, answer, mode="summary",
                                     history_turns=1)
    p.turn("my name is Ken")                   # gate: name
    m.wait(10)
    p.turn("let's guess a number")             # gate: game
    m.wait(10)
    assert m.notes.summary == []
    p.turn("it's seven")                       # turn 1 leaves the window
    m.wait(10)
    p.turn("hello")                            # and turn 2
    m.wait(10)
    prompts = [c[1] for c in cons.calls if "EXCHANGES DROPPED" in c[1]]
    assert prompts and "viewer: my name is Ken" in prompts[0]
    assert m.notes.summary == ["a number game was played"]
    assert "earlier: a number game was played" in m.block()
    assert '"summary"' in cons.calls[0][0]
    # the next request carries it
    p.turn("still here")
    assert "earlier: a number game was played" in student.requests[-1][-1]["content"]


def test_notes_mode_never_asks_for_or_keeps_a_summary(tmp_path):
    ans = notes_json(facts=["a"], summary=["earlier on stream: x"])
    p, hub, student, cons, m = build(tmp_path, ans, history_turns=1)
    for t in ("my name is Ken", "hello", "it is fine", "ok then"):
        p.turn(t)
        m.wait(10)
    assert m.notes.summary == [] and "earlier" not in m.block()
    assert all("DROPPED" not in c[1] for c in cons.calls)


def test_the_summary_is_capped(tmp_path):
    ans = notes_json(summary=["s" * 150, "t" * 150, "u" * 150])
    p, hub, student, cons, m = build(tmp_path, ans, mode="summary")
    p.turn("remember that")
    m.wait(10)
    assert sum(map(len, m.notes.summary)) <= mem.SUMMARY_MAX
    assert len(m.block()) <= 800


def test_a_failed_fold_is_retried_with_the_same_exchanges(tmp_path):
    answers = ["not json", notes_json(summary=["earlier on stream: hi"])]
    p, hub, student, cons, m = build(tmp_path, answers, mode="summary",
                                     history_turns=1)
    for t in ("hello one", "hello two", "hello three", "hello four"):
        p.turn(t)
        m.wait(10)
    assert m.notes.summary == ["earlier on stream: hi"]
    assert "viewer: hello one" in cons.calls[1][1]


# -------------------------------------------------------------- recorder

def test_each_consolidation_is_a_memory_record(tmp_path):
    p, hub, student, cons, m = build(
        tmp_path, [notes_json(facts=["viewer Ken"]),
                   notes_json(facts=["viewer Ken", "likes tea"])])
    p.turn("my name is Ken")
    m.wait(10)
    p.turn("remember that I like tea")
    m.wait(10)
    mrows = [r for r in p.recorder.rows if r.get("kind") == "memory"]
    assert len(mrows) == 2
    r = mrows[1]
    assert r["ok"] and r["turn"] == 2 and "remember" in r["reasons"]
    assert r["before"]["facts"] == ["viewer Ken"]
    assert r["after"]["facts"] == ["viewer Ken", "likes tea"]
    assert r["diff"] == {"facts": {"added": ["likes tea"], "removed": []}}
    json.dumps(r, ensure_ascii=False)
    # turn records are the Performer's own, as before
    assert [x["turn"] for x in p.recorder.rows if "user" in x] == [1, 2]


def test_memory_rows_do_not_break_the_checkers(tmp_path):
    path = tmp_path / "turns.jsonl"
    rows = [{"kind": "memory", "turn": 1, "ok": True},
            {"turn": 1, "user": "hi", "reply": "Hello!"},
            {"kind": "memory", "turn": 1, "ok": False}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ((label, recs),) = ck.load(str(path), "Mao")
    assert len(recs) == 1 and recs[0]["user"] == "hi"


def test_recorder_writes_memory_rows_and_memory_json(tmp_path):
    rec = lv.Recorder(str(tmp_path / "rec"), 30, video=False)
    m = mem.Memory("notes", 800, Consolidator(notes_json(facts=["a"])))
    p, hub, *_ = _performer(tmp_path, "", backend=Student(), memory=m)
    p.recorder = rec
    m.path = str(tmp_path / "rec" / "memory.json")
    p.turn("remember that I like tea")
    m.wait(10)
    rec.turns.flush()
    lines = [json.loads(x) for x in
             (tmp_path / "rec" / "turns.jsonl").read_text().splitlines()]
    assert {x.get("kind") for x in lines} == {None, "memory"}
    saved = json.loads((tmp_path / "rec" / "memory.json").read_text())
    assert saved["facts"] == ["a"]
    rec.turns.close()


# --------------------------------------------------------------- the game

def test_number_game_notes_keep_15_too_big_after_it_is_trimmed(tmp_path):
    game1 = "number game, 1-20, she guesses"
    game2 = game1 + ": 10 too small"
    game3 = game2 + ", 15 too big"
    answers = [notes_json(game=[game1]), notes_json(game=[game2]),
               notes_json(game=[game3])]
    p, hub, student, cons, m = build(tmp_path, answers, history_turns=2)
    script = ["let's play a number game, 1 to 20", "10 is too small",
              "15 is too big", "hmm", "so?", "what now", "go on"]
    for line in script:
        p.turn(line)
        m.wait(10)
    # By now the exchange that said "15 is too big" has left the history ...
    last = student.requests[-1]
    assert not any("15" in x["content"] for x in last[:-1])
    assert len(last) == 2 * 2 + 1
    # ... and the notes still carry it, on the latest user message only.
    assert last[-1]["content"].endswith(
        f"(stream notes: game: {game3})")
    assert "15 too big" in last[-1]["content"]
    assert all("stream notes" not in x["content"] for x in last[:-1])
    # Small talk did not call the consolidator again (3 game turns only).
    assert len(cons.calls) == 3
    # Without memory the same script forgets it.
    q = Student()
    (tmp_path / "off").mkdir()
    p2, *_ = _performer(tmp_path / "off", "", backend=q, history_turns=2)
    for line in script:
        p2.turn(line)
    assert not any("15" in x["content"] for x in q.requests[-1])
    assert "stream notes" not in q.requests[-1][-1]["content"]


def test_the_prompts_and_flags():
    s = mem.system_prompt(True, 800)
    assert "800" in s and "summary" in s and "{" in s
    assert "summary" not in mem.system_prompt(False, 800).split(
        "Rules:")[0]
    import argparse
    ap = argparse.ArgumentParser()
    mem.add_args(ap)
    a = ap.parse_args([])
    assert (a.memory, a.memory_backend, a.memory_chars) == \
        ("off", "deepseek", 800)
    a = ap.parse_args(["--memory", "summary", "--memory-chars", "500"])
    m = mem.from_args(a, backend=object())
    assert m.mode == "summary" and m.chars == 500


def test_the_prompt_asks_for_roles_the_next_move_and_lasting_facts():
    from flow import memory as mm
    for summary in (False, True):
        p = mm.system_prompt(summary)
        # who did what, by role (live: "you won" after she guessed 67)
        assert 'never "you" or "we"' in p
        # the next move spelled out (live: "timber" -> "start with t")
        assert "NEXT MOVE" in p and "must start with R" in p
        # no one-off requests as facts (live: "viewer asked her to repeat")
        assert "Never what someone asked for" in p
    assert "who won, by role" in mm.system_prompt(True)
