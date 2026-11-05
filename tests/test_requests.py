"""
An explicit request for a hand action wins over the bust rule (beats2 rev 2):
the prompt sentences, the export flag skipped_request for the turns where the
teacher answered with the head only, and a dataset run resumed after the
prompt changed (both prompts side by side in avatars.json, the export and the
short student prompt reading each). No network: mocked models.
"""

from __future__ import annotations

import json
import os

import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import dataset as ds
from flow import framing as fr
from flow import pose as pz
from tests.test_beats import FACES, RIGS
from tests.test_dataset import CharLLM, UserLLM

HANDS = ('Sure! [pose {"hand.R.y": 0.6, "hand.R.x": 0.5, "dur": 0.4}] '
         'There.')


def _bm(name):
    return fb.load(next(p for p in RIGS if os.path.basename(p)
                        .startswith(name)))


# ------------------------------------------------------------ the prompt

def test_camera_skip_applies_to_unasked_moves_in_beats2_only():
    bm = _bm("Hiyori")
    old = pz.camera_rule(bm)
    new = pz.camera_rule(bm, True)
    assert "a hand move below y -1.00 is not seen: skip it" in old
    assert "unasked hand move below y -1.00 is not seen: skip it" in new
    assert "A hand move the user asks for is not skipped" in new
    assert new != old
    # the beats prompt keeps its bytes
    assert old in pz.system_prompt(bm, "X.", prompt="beats")
    assert "unasked" not in pz.system_prompt(bm, "X.", prompt="beats")


def test_beats2_makes_explicit_requests_win():
    hiyori = pz.system_prompt(_bm("Hiyori"), "X.", prompt="beats2")
    assert "unasked hand move below" in hiyori
    i = hiyori.index("- When the user asks you for a hand or arm action")
    rule = hiyori[i:hiyori.index("\n", i)]
    assert "always do it with your hand in that reply" in rule
    assert "pointing is a hand pointing (your eyes may follow it, never " \
        "instead of it), a peace sign is the hand up" in rule
    assert "In bust raise the hand into the picture, above y -1.00" in rule
    assert "write [frame full] first" in rule
    # after the budget it overrides
    assert i > hiyori.index("Motion budget")
    # the first r3 wording stays rebuildable
    old = pz.system_prompt(_bm("Hiyori"), "X.", prompt="beats2", rev=1)
    assert "hand or arm action" not in old and "unasked" not in old
    assert old != hiyori


def test_a_limited_rig_does_its_closest_and_a_handless_body_says_so(
        monkeypatch):
    lafei = pz.request_rule(_bm("lafei"))
    assert "Your arms are limited: do the closest your arm can" in lafei
    hiyori = pz.request_rule(_bm("Hiyori"))
    assert hiyori.startswith("- When the user asks you")
    orig = fr.budget
    monkeypatch.setattr(fr, "budget", lambda bm, *a, **k: {
        **orig(_bm("Hiyori")), "hand_moves": {"bust": 0, "full": 0}})
    text = pz.request_rule(None)
    assert text.startswith("- You cannot move your hands")
    assert "say so naturally" in text and "[frame full]" not in text


def test_short_prompt_carries_the_request_rule():
    bm = _bm("Hiyori")
    short = pz.system_prompt(bm, "X.", prompt="short")
    assert "unasked hand move below" in short
    assert "- Asked for a hand action (point, wave, peace sign, clap): " \
        "always do it with your hand, whatever the budget; in bust raise " \
        "it above y -1.00" in short


# ---------------------------------------------------------- the flag

def _run(tmp_path, reply, sessions=4, turns=3, avatars="Hiyori,seed-san",
         prompt="beats2", out=None, concurrency="1"):
    out = out or tmp_path / "run"

    def make(role, av):
        return CharLLM([reply]) if role == "character" else UserLLM()
    opts = ds.parse_opts(["gen", "--avatars", avatars, "--live2d-lite",
                          "--sessions", str(sessions), "--turns", str(turns),
                          "--out", str(out), "--personas", "cheerful",
                          "--teacher-prompt", prompt, "--concurrency",
                          concurrency, "--quiet-share", "0"])
    ds.generate(opts, backend_factory=make, log=lambda s: None)
    return str(out), ds.iter_records(str(out))


@pytest.fixture(scope="module")
def faces_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("req"), FACES, turns=5)


def test_turns_that_asked_and_got_no_hand_are_flagged(faces_run):
    out, recs = faces_run
    req = dx.request_bodies(out)
    assert req == {"Hiyori": True, "seed-san": True}
    notes = dx.annotate(recs, requests=req)
    flagged = {r["id"]: r["user"] for r in recs
               if "skipped_request" in notes[r["id"]][0]}
    assert flagged and all(dx.skipped_request(r, req) for r in recs
                           if r["id"] in flagged)
    users = set(flagged.values())
    assert any("wave" in u.lower() for u in users)
    assert any("Point at" in u for u in users)
    for r in recs:
        f, kept = notes[r["id"]]
        if "skipped_request" in f:
            assert not kept
        else:                       # the greeting, the invitation-like ones
            assert not dx.skipped_request(r, req)
    # off without the table, and for a body whose hands reach nothing
    off = dx.annotate(recs)
    assert not any("skipped_request" in f for f, _ in off.values())
    none = dx.annotate(recs, requests={"Hiyori": False, "seed-san": False})
    assert not any("skipped_request" in f for f, _ in none.values())


def test_a_hand_in_the_reply_is_no_skip(tmp_path):
    out, recs = _run(tmp_path, HANDS, sessions=2)
    req = dx.request_bodies(out)
    assert not any(dx.skipped_request(r, req) for r in recs)
    assert dx.reply_moves_hand('[move {"name": "wave"}] Hi')
    assert dx.reply_moves_hand("[gesture:wave_a] Hi")
    assert not dx.reply_moves_hand(FACES)


def test_session_export_is_cut_before_the_skipped_turn(faces_run, tmp_path,
                                                       capsys):
    out, recs = faces_run
    req = dx.request_bodies(out)
    on = dx.annotate(recs, requests=req)
    off = dx.annotate(recs)
    sess_on = dx.session_turns(recs, on)
    sess_off = dx.session_turns(recs, off)
    assert sum(map(len, sess_on.values())) < sum(map(len, sess_off.values()))
    for sid, turns in sess_on.items():
        assert all("skipped_request" not in on[r["id"]][0] for r in turns)
        # the turns before the first skip are all there
        first = next((r["turn"] for r in sorted(
            (x for x in recs if x["session"] == sid),
            key=lambda x: x["turn"]) if "skipped_request" in on[r["id"]][0]),
            None)
        if first:
            assert [r["turn"] for r in turns] == list(range(1, first))
    # per turn: the turns that would show the skip in their history go too
    ex = dx.sft_examples(out, recs, on, per="turn", meta=True)
    ids = {e["meta"]["id"] for e in ex}
    for r in recs:
        if "skipped_request" in on[r["id"]][0]:
            assert r["id"] not in ids
            assert not any(x["session"] == r["session"]
                           and x["turn"] > r["turn"] and x["id"] in ids
                           for x in recs)
    # stats and sft count them; the default follows the prompt
    assert dx.default_request_filter(out) is True
    assert dx.main(["stats", out, "--keep-lite"]) == 0
    text = capsys.readouterr().out
    n = sum("skipped_request" in f for f, _ in on.values())
    assert f"requests: {n} turns dropped (skipped hand request)" in text
    dst = tmp_path / "a.jsonl"
    assert dx.main(["sft", out, "--out", str(dst), "--keep-lite"]) == 0
    assert f"requests: {n} turns dropped" in capsys.readouterr().out
    kept_on = len(dst.read_text().splitlines())
    dx.main(["sft", out, "--out", str(dst), "--keep-lite",
             "--no-request-filter"])
    capsys.readouterr()
    assert len(dst.read_text().splitlines()) == kept_on
    turns_on = sum(len(json.loads(x)["messages"]) for x in
                   dst.read_text().splitlines())
    dx.main(["sft", out, "--out", str(dst), "--keep-lite", "--request-filter"])
    capsys.readouterr()
    assert sum(len(json.loads(x)["messages"]) for x in
               dst.read_text().splitlines()) < turns_on


def test_older_data_is_exported_as_before(tmp_path, capsys):
    out, recs = _run(tmp_path, FACES, sessions=2, prompt="beats")
    assert dx.default_request_filter(out) is False
    notes = dx.annotate(recs)
    assert dx.main(["stats", out]) == 0
    assert "requests:" not in capsys.readouterr().out
    out1, _ = _run(tmp_path / "v1", FACES, sessions=2, prompt="v1")
    assert dx.default_request_filter(out1) is False
    assert all("skipped_request" not in f for f, _ in notes.values())


# ------------------------------------------- resume with a changed prompt

def test_resume_after_the_prompt_changed_keeps_both(tmp_path, monkeypatch,
                                                    capsys):
    out = tmp_path / "r3"
    monkeypatch.setattr(pz, "BEATS2_REV", 1)
    _out, first = _run(tmp_path, FACES, sessions=3, out=out)
    old_prompts = dict(dx.load_registry(str(out))["prompts"])
    monkeypatch.setattr(pz, "BEATS2_REV", 2)
    logs = []

    def make(role, av):
        return CharLLM([FACES]) if role == "character" else UserLLM()
    opts = ds.parse_opts(["gen", "--avatars", "Hiyori,seed-san",
                          "--live2d-lite", "--sessions", "6", "--turns", "3",
                          "--out", str(out), "--personas", "cheerful",
                          "--teacher-prompt", "beats2", "--concurrency", "1",
                          "--quiet-share", "0"])
    res = ds.generate(opts, backend_factory=make, log=logs.append)
    assert res["failed"] == 0
    assert any("prompt changed" in x or "prompt_sha" in x for x in logs)
    recs = ds.iter_records(str(out))
    reg = dx.load_registry(str(out))
    # the old sessions are untouched, the new ones carry the new sha
    assert {r["id"] for r in first} < {r["id"] for r in recs}
    shas = {r["avatar"]["name"]: set() for r in recs}
    for r in recs:
        shas[r["avatar"]["name"]].add(r["avatar"]["prompt_sha"])
        assert r["avatar"]["prompt_sha"] in reg["prompts"]
    old_shas = set(old_prompts)
    for r in first:
        assert r["avatar"]["prompt_sha"] in old_shas
    for name, s in shas.items():
        assert s & old_shas and s - old_shas, name
    for name, av in reg["avatars"].items():
        assert av["earlier"], name
    new = [t for k, t in reg["prompts"].items() if k not in old_shas]
    assert new and all("always do it with your hand in that reply" in t
                       for t in new)
    assert all("hand or arm action" not in t for t in old_prompts.values())
    # the export reads both: the teacher prompts as written ...
    fix, _ = dx.correct_misses(recs)
    notes = dx.annotate(recs, fix=fix, keep_lite=True)
    teacher = dx.sft_examples(str(out), recs, notes, meta=True, fix=fix)
    used = {e["messages"][0]["content"] for e in teacher}
    assert used & set(old_prompts.values())
    assert used & {t for k, t in reg["prompts"].items() if k not in old_shas}
    # ... and the short student prompt, rebuilt for each version
    short = dx.sft_examples(str(out), recs, notes, meta=True, fix=fix,
                            student="short")
    assert len(short) == len(teacher)
    assert len({e["messages"][0]["content"] for e in short}) \
        >= len({e["meta"]["avatar"] for e in short})
    for r in recs:
        teacher_sys = reg["prompts"][r["avatar"]["prompt_sha"]]
        dx.student_system(teacher_sys, r, "short")
    # a prompt that does not rebuild is still refused
    with pytest.raises(SystemExit, match="byte for byte"):
        dx.student_system(teacher_sys + "x", recs[-1], "short")
    # the CLI (stats, sft with the short prompt) and the held-out replay
    assert dx.main(["stats", str(out), "--keep-lite"]) == 0
    dst = tmp_path / "s.jsonl"
    assert dx.main(["sft", str(out), "--out", str(dst), "--keep-lite",
                    "--student-prompt", "short"]) == 0
    capsys.readouterr()
    # the held-out replay: every session reads the prompt of its own
    # records (old or new), rebuilt as the short student prompt on request
    from flow import director as dr
    from flow.train import evalsft
    sess = {}
    for r in recs:
        sess.setdefault(r["session"], []).append(r)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"eval_sessions": sorted(sess),
                                 "train_sessions": []}))
    made = []

    def backend(a, log=print):
        made.append(CharLLM([FACES]))
        return made[-1]
    monkeypatch.setattr(dr, "make_backend", backend)
    for flag, name in (([], "teacher"), (["--student-prompt", "short"],
                                         "short")):
        made.clear()
        o = evalsft.parse(["run", "--data", str(out), "--split", str(split),
                           "--out", str(tmp_path / "ev"), "--name", name,
                           *flag])
        evalsft.run(o, log=lambda s: None)
        assert len(made) == len(sess)
        for sid, be in zip(sorted(sess), made):
            rs = sorted(sess[sid], key=lambda r: r["turn"])
            want = reg["prompts"][rs[0]["avatar"]["prompt_sha"]]
            if flag:
                want = dx.student_system(want, rs[0], "short")
            assert {x[0] for x in be.sent} == {want}, (name, sid)
    sent = {s_ for be in made for s_ in (x[0] for x in be.sent)}
    assert len(sent) >= 2
