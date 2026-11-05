"""
A small beats3 pilot with fake models (no DeepSeek): the notes and reminders
tail in the records and the SFT export, a game with its notes, bursts of
chat lines, the history window, the export defaults of a beats3 dataset, and
the student prompts rebuilt from it. The real-chat flags are in
test_realdata.py.
"""

from __future__ import annotations

import copy
import json

import pytest

from flow import dataexport as dx
from flow import dataset as ds
from flow import pose as pz

FACES = ('[pose {"brow.L.lift": 0.6, "brow.R.lift": 0.6, "head.tilt": 5, '
         '"dur": 0.2}] I once lost a ferry ticket in Osaka. [pose '
         '{"mouth.form": 0.8, "head.pitch": 3}] Funny, right?')


class Char:
    def __init__(self):
        self.sent, self.usage, self.stop_reason = [], {}, None
        self.model = "fake-char"

    def stream(self, system, messages):
        self.sent.append((system, copy.deepcopy(messages)))
        yield FACES
        self.stop_reason = "stop"
        self.usage = {"prompt_tokens": 100, "completion_tokens": 20}


class Viewer:
    LINES = ["Hello there, long time no see!", "Let's play a number game!",
             "Is it 25?", "Maybe 40, then.", "How about 37?",
             "hi from Brazil!\nwhat did you eat?\nlol that laugh",
             "It is not Kyoto, it is Kobe.", "My name is Mika, by the way."]

    def __init__(self):
        self.k, self.usage, self.stop_reason = 0, {}, None
        self.model, self.sent = "fake-user", []

    def stream(self, system, messages):
        self.sent.append((system, messages[-1]["content"]))
        line = self.LINES[self.k % len(self.LINES)]
        self.k += 1
        self.usage = {"prompt_tokens": 50, "completion_tokens": 10}
        yield line


def _gen(tmp_path, *extra, sessions=3, turns=8, avatars="Hiyori"):
    out = tmp_path / "run"
    made = {"character": [], "user": []}

    def make(role, av):
        b = Char() if role == "character" else Viewer()
        made[role].append(b)
        return b
    opts = ds.parse_opts(["gen", "--avatars", avatars, "--live2d-lite",
                          "--sessions", str(sessions), "--turns", str(turns),
                          "--out", str(out), "--personas", "cheerful",
                          "--backoff", "0.01", "--quiet-share", "0",
                          "--prompt", "beats3", "--seed", "3", *extra])
    res = ds.generate(opts, backend_factory=make, log=lambda s: None)
    assert res["failed"] == 0
    return str(out), ds.iter_records(str(out)), made


def test_prompt_flag_alias_and_defaults():
    o = ds.parse_opts(["gen", "--prompt", "beats3"])
    assert o.teacher_prompt == "beats3"
    assert ds.parse_opts(["gen"]).teacher_prompt == "beats2"
    o = ds.parse_opts(["gen"])
    assert (o.reminders, o.notes, o.game_share, o.history_turns,
            o.real_viewer, o.style_seeds) == ("off", "off", None, 0, None,
                                              None)
    with pytest.raises(SystemExit):
        ds.parse_opts(["gen", "--reminders", "sometimes"])


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    return _gen(tmp_path_factory.mktemp("p3"), "--notes", "auto",
                "--reminders", "auto", "--game-share", "1",
                "--game-kinds", "number", "--history-turns", "3")


def test_the_pilot_writes_beats3_records(pilot):
    out, recs, made = pilot
    assert len(recs) == 24
    reg = dx.load_registry(out)
    for r in recs:
        assert r["avatar"]["prompt_sha"] in reg["prompts"]
        t = reg["prompts"][r["avatar"]["prompt_sha"]]
        assert pz.prompt_family(t) == "beats3" and pz.BIBLE_HEAD in t
        assert 0 <= r["avatar"]["variant"] < 6
        assert len(r["history"]) == min(r["turn"] - 1, 3)   # the window
    # the viewer never saw the bible, and the character did
    for b in made["user"]:
        assert all("Backstory" not in s for s, _ in b.sent)
    assert all("Backstory" in s for b in made["character"]
               for s, _ in b.sent)
    # a count of the viewer modes drawn
    topics = {r["topic"] for r in recs}
    assert "game_number" in topics


def test_notes_ride_the_latest_turn_only(pilot):
    out, recs, made = pilot
    game = [r for r in recs if r.get("game")]
    assert game and all(r["topic"] == "game_number" for r in game)
    with_tail = [r for r in recs if r.get("tail")]
    assert with_tail
    for r in with_tail:
        # the record keeps user_content plain; the block is its own field
        assert "(stream notes" not in r["user_content"]
        assert r["tail"].startswith("(stream notes: ") or \
            r["tail"].startswith("(reminders: ")
        assert all("(stream notes" not in h["user_content"]
                   for h in r["history"])
    # byte for byte what the character got: the last user message
    sent = {}
    for b in made["character"]:
        for system, msgs in b.sent:
            # revision 2 ends the teacher's copy with its length line
            # (flow.style.length_line), which is not the record's
            last = msgs[-1]["content"].split("\n(length for this reply:")[0]
            sent[last] = msgs
    for r in with_tail:
        want = r["user_content"] + "\n" + r["tail"]
        assert want in sent, r["id"]
        msgs = sent[want]
        assert all("(stream notes" not in m["content"] for m in msgs[:-1])
    # a game turn's notes hold her secret; later ones judge the guesses
    g = [r for r in game if r.get("tail")]
    assert g and all("her secret is " in r["tail"] and r["notes"]["game"]
                     for r in g)
    assert any("guesses: " in r["tail"] and "too " in r["tail"] for r in g)


def test_reminders_use_the_live_nodes(pilot):
    out, recs, _ = pilot
    # a game is on: the game node (needs the notes) is among the reminders
    ids = {i for r in recs for i in r.get("rules", ())}
    assert "game" in ids or any("(reminders:" in r.get("tail", "")
                                for r in recs)
    for r in recs:
        if r.get("tail"):
            assert r["tail"].count("(stream notes:") <= 1


def test_the_export_appends_the_tail_to_that_turn_only(pilot):
    out, recs, _ = pilot
    fix, _ = dx.correct_misses(recs)
    notes = dx.annotate(recs, fix=fix, keep_lite=True, lang_filter=True)
    per_turn = dx.sft_examples(out, recs, notes, per="turn", meta=True,
                               fix=fix)
    n = 0
    for ex in per_turn:
        rid = ex["meta"]["id"]
        r = next(x for x in recs if x["id"] == rid)
        msgs = ex["messages"]
        assert msgs[-2]["content"] == r["user_content"] + (
            "\n" + r["tail"] if r.get("tail") else "")
        assert all("(stream notes" not in m["content"] for m in msgs[:-2])
        n += bool(r.get("tail"))
    assert n
    per_session = dx.sft_examples(out, recs, notes, per="session", fix=fix)
    assert any("(stream notes" in m["content"] for ex in per_session
               for m in ex["messages"] if m["role"] == "user")


def test_a_beats3_dataset_exports_with_the_beats2_defaults(pilot):
    out, recs, _ = pilot
    assert dx.default_max_beats(out) == dx.BEATS_MAX
    assert dx.default_max_hands(out) == dx.MAX_HANDS
    assert dx.default_lang_filter(out) is True
    assert dx.default_request_filter(out) is True


def test_student_prompts_rebuild_from_beats3_data(pilot):
    out, recs, _ = pilot
    fix, _ = dx.correct_misses(recs)
    notes = dx.annotate(recs, fix=fix, keep_lite=True, lang_filter=True)
    cfg = ds.load_config(None)
    persona = cfg["personas"][0]
    teacher = dx.sft_examples(out, recs, notes, meta=True, fix=fix)
    short = dx.sft_examples(out, recs, notes, meta=True, fix=fix,
                            student="short")
    sb = dx.sft_examples(out, recs, notes, meta=True, fix=fix,
                         student="short+bible")
    assert len(teacher) == len(short) == len(sb) >= 1
    bible = ds.bible_text(persona).splitlines()[0]
    for t, s, b in zip(teacher, short, sb):
        assert bible in t["messages"][0]["content"]
        assert bible not in s["messages"][0]["content"]
        assert bible in b["messages"][0]["content"]
        assert s["messages"][1:] == t["messages"][1:]
        # the serving prompt: what flow.director builds from the persona
        # text (with the bible behind its separator) for short / short+bible
        spec = ds.avatar_spec("Hiyori", lite=True)
        o = ds.parse_opts(["gen", "--prompt", "beats3"])
        ptext = ds.persona_text(cfg, persona, "Hiyori", bible=True,
                                rev=ds.beats3_rev(o))
        a = ds.director_args(spec, o, ptext)
        bm = ds.fb.load(spec["body"])
        ml, _n = ds.dr.move_menu(a, bm)
        assert s["messages"][0]["content"] == pz.system_prompt(
            bm, ptext, False, "", ml, "short")
        assert b["messages"][0]["content"] == pz.system_prompt(
            bm, ptext, False, "", ml, "short+bible")


def test_chat_bursts_are_several_lines_in_one_message(tmp_path):
    out, recs, _ = _gen(tmp_path, "--game-share", "0", sessions=6, turns=10)
    bursts = [r for r in recs if r["topic"] == "chat_burst"]
    assert bursts
    assert any("\n" in r["user"] for r in bursts)
    for r in bursts:
        assert r["user_content"].startswith(r["user"] + "\n(your body now")
        assert len(r["user"].split("\n")) <= 5
    assert not any(r.get("tail") for r in recs)        # flags off: no tail
    assert not any(r.get("game") for r in recs)


def test_beats2_sessions_do_not_change(tmp_path):
    """The new flags are off for beats2: no tail, no extras, the same
    topics (those of beats2 only)."""
    out = tmp_path / "b2"

    def make(role, av):
        return Char() if role == "character" else Viewer()
    opts = ds.parse_opts(["gen", "--avatars", "Hiyori", "--live2d-lite",
                          "--sessions", "3", "--turns", "6", "--out",
                          str(out), "--personas", "cheerful", "--backoff",
                          "0.01", "--quiet-share", "0"])
    ds.generate(opts, backend_factory=make, log=lambda s: None)
    recs = ds.iter_records(str(out))
    new = {"ask_her", "story", "disagree", "correct", "callback",
           "chat_burst"}
    assert not ({r["topic"] for r in recs} & new)
    assert not any(k in r for r in recs
                   for k in ("tail", "notes", "rules", "game", "real"))
    assert all(pz.BIBLE_HEAD not in t for t in
               dx.load_registry(str(out))["prompts"].values())


def test_notes_and_reminders_need_a_beats_prompt(tmp_path):
    with pytest.raises(SystemExit, match="beats"):
        ds.generate(ds.parse_opts(["gen", "--avatars", "Hiyori",
                                   "--live2d-lite", "--prompt", "v1",
                                   "--reminders", "auto", "--out",
                                   str(tmp_path / "x")]),
                    backend_factory=lambda *a: None, log=lambda s: None)
    with pytest.raises(SystemExit, match="beats3"):
        ds.generate(ds.parse_opts(["gen", "--avatars", "Hiyori",
                                   "--live2d-lite", "--style-seeds", "x",
                                   "--out", str(tmp_path / "y")]),
                    backend_factory=lambda *a: None, log=lambda s: None)


def test_evalsft_sends_the_recorded_tail_on_the_latest_turn_only(pilot,
                                                                 tmp_path):
    """A student trained with the notes and reminders is evaluated with
    them: the held-out turn's user message ends with the record's tail, the
    history is plain, and --prompt-cond full+rem adds no second reminders
    block."""
    from flow import director as dr
    from flow.train import evalsft
    out, recs, _ = pilot
    sessions = sorted({r["session"] for r in recs})
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"eval_sessions": sessions,
                                 "train_sessions": []}))
    calls = []

    class Fake:
        model, stop_reason, usage = "fake", "stop", {}

        def stream(self, system, messages):
            calls.append([dict(m) for m in messages])
            yield FACES
    real = dr.make_backend
    try:
        dr.make_backend = lambda a, log=print: Fake()
        for cond in ("full", "full+rem"):
            calls.clear()
            argv = ["run", "--data", out, "--split", str(split), "--out",
                    str(tmp_path / "ev"), "--name", "student"]
            if cond != "full":
                argv += ["--prompt-cond", cond]
            evalsft.run(evalsft.parse(argv), log=lambda s: None)
            by_content = {m[-1]["content"]: m for m in calls}
            tails = [r for r in recs if r.get("tail")]
            assert tails
            for r in tails:
                want = r["user_content"] + "\n" + r["tail"]
                assert want in by_content, (cond, r["id"])
                msgs = by_content[want]
                assert all("(stream notes" not in m["content"]
                           for m in msgs[:-1])
                assert msgs[-1]["content"].count("(reminders:") <= 1
    finally:
        dr.make_backend = real
