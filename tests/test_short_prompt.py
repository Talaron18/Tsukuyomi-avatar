"""
The student prompt "short" (flow.pose.short_prompt): far shorter than the
beats2 teacher prompt, keeping the body description, the motion budget and
the camera rule, with no dense or placement example; never a teacher's
prompt; rebuilt by the SFT export from the avatar's body file and persona
to the bytes the director serves. No network: the dataset runs are mocked
models.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import dataset as ds
from flow import director as dr
from flow import pose as pz
from tests.test_beats import FACES, RIGS
from tests.test_beats2 import face_bm  # noqa: F401 - the fixture
from tests.test_dataexport import _run
from tests.test_dataset import ROOT

PERSONA = "Hiyori, a cheerful girl who loves to chat."


def _bodies(face):
    out = [(os.path.basename(p).split(".")[0], fb.load(p)) for p in RIGS]
    out.append(("face", face))
    return out


def _menu(bm):
    return dr.move_menu(SimpleNamespace(body_moves=True), bm)[0]


# ----------------------------------------------------------- the prompt

def test_short_is_much_shorter_and_keeps_what_varies(face_bm):  # noqa: F811
    for name, bm in _bodies(face_bm):
        m = _menu(bm)
        b2 = pz.system_prompt(bm, PERSONA, False, "", m, "beats2")
        sh = pz.system_prompt(bm, PERSONA, False, "", m, "short")
        assert len(sh) < 0.5 * len(b2), (name, len(sh), len(b2))
        assert sh.startswith("You are " + PERSONA)
        # the body description in full, the budget and the camera rule
        assert pz.describe_body(bm) in sh, name
        assert pz.budget_line(bm, True) in sh, name
        assert pz.camera_rule(bm, True) in sh, name
        assert pz.request_rule(bm, True) in sh, name
        # no dense example, no placement examples, no long beats rule
        for v in pz.dense_variants(bm):
            assert v not in sh
        assert "Examples of placement" not in sh
        assert "A beat is a quick change of" not in sh
        assert "Vary your moves" not in sh
        assert sh.count("[pose {") < 6
        # the marker syntax, compactly, with the variables this body has
        have = set(pz.offered(bm))
        ex = pz._pose_example(bm)
        assert ex in sh
        assert set(json.loads(ex[len("[pose "):-1])) - {"dur"} <= have
        assert "[blink]" in sh and "[pause:0.5]" in sh
        assert ("[frame bust] or [frame full]" in sh) == pz.fr.can_frame(bm)
        # the user turn's lines
        for s in ("(your body now: ...)", "(camera: ...)",
                  "(reply in: English)", pz.QUIET_CHAT):
            assert s in sh, (name, s)
        # a student prompt is byte-stable and has no teacher-only rules
        assert sh == pz.system_prompt(bm, PERSONA, False, "", m, "short")
        assert dx.BEATS_MARK not in sh and pz.OPEN_RULE not in sh


def test_short_moves_gestures_and_looks(face_bm):  # noqa: F811
    bm = fb.load(RIGS[0])
    gl = dr.gesture_lines({"wave_a": "wave_a"}, {"wave_a": "a big wave"})
    sh = pz.system_prompt(bm, PERSONA, False, gl, _menu(bm), "short")
    assert gl in sh and "NAME one of: sway" in sh
    assert '"head.yaw" and "gaze.x"' in sh and "negative" in sh
    nomove = pz.system_prompt(bm, PERSONA, False, "", "", "short")
    assert "[move {" not in nomove
    img = pz.system_prompt(bm, PERSONA, True, "", "", "short")
    assert "a picture of how you look" in img


def test_short_is_a_student_prompt_that_parses_like_beats2(face_bm):  # noqa: F811
    assert "short" not in pz.PROMPTS
    assert pz.STUDENT_PROMPTS == ("short", "short+bible")
    assert pz.beats_like("short")
    for name, bm in _bodies(face_bm):
        for fram in ("bust", "full"):
            for text, lang in (("Hello!", None), ("こんにちは", None),
                               ("(chat is quiet)", "你好")):
                assert pz.turn_tail(bm, fram, text, "short", lang) \
                    == pz.turn_tail(bm, fram, text, "beats2", lang)
    with pytest.raises(ValueError):
        pz.system_prompt(fb.load(RIGS[0]), "X.", prompt="short2")


def test_the_director_and_live_take_short():
    ap = dr.argparse.ArgumentParser()
    dr.add_args(ap)
    assert ap.parse_args(["--body-prompt", "short"]).body_prompt == "short"
    with pytest.raises(SystemExit):
        ap.parse_args(["--body-prompt", "short2"])
    # flow.live builds its parser from the director's (add_args)
    from flow import live
    assert "dr.add_args(ap)" in open(live.__file__, encoding="utf-8").read()


# ------------------------------------------------------ never the teacher

def test_the_dataset_refuses_short_as_the_teacher(tmp_path):
    with pytest.raises(SystemExit):
        ds.parse_opts(["gen", "--teacher-prompt", "short"])
    opts = ds.parse_opts(["gen", "--avatars", "Hiyori", "--live2d-lite",
                          "--out", str(tmp_path / "x")])
    opts.teacher_prompt = "short"
    with pytest.raises(SystemExit, match="student prompt"):
        ds.generate(opts, log=lambda s: None)
    with pytest.raises(SystemExit, match="student prompt"):
        pz.check_teacher_prompt("short")
    assert pz.check_teacher_prompt("beats2") == "beats2"


# ----------------------------------------------- export, training, serving

@pytest.fixture(scope="module")
def beats2_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("short"), FACES,
                avatars="seed-san,Hiyori,lafei", sessions=6, turns=3,
                prompt="beats2")


def _examples(out, recs, student):
    fix, _ = dx.correct_misses(recs)
    notes = dx.annotate(recs, fix=fix, keep_lite=True, lang_filter=True)
    return dx.sft_examples(out, recs, notes, meta=True, fix=fix,
                           student=student)


def test_export_rebuilds_the_prompt_the_director_serves(beats2_run):
    out, recs = beats2_run
    teacher = _examples(out, recs, "teacher")
    short = _examples(out, recs, "short")
    assert len(teacher) == len(short) >= 3
    cfg = ds.load_config(None)
    pid = {p["id"]: p for p in cfg["personas"]}
    seen = set()
    for t, s in zip(teacher, short):
        name, persona = s["meta"]["avatar"], s["meta"]["persona"]
        seen.add(name)
        # only the system prompt differs: user turns and replies are as
        # exported
        assert s["messages"][1:] == t["messages"][1:]
        assert s["messages"][0]["content"] != t["messages"][0]["content"]
        # the serving prompt: flow.director's arguments for this avatar
        # with --body-prompt short, the persona the dataset used
        spec = ds.avatar_spec(name, lite=True)
        opts = ds.parse_opts(["gen"])
        opts.teacher_prompt = "beats2"
        ptext = ds.persona_text(cfg, pid[persona], name)
        a = ds.director_args(spec, opts, ptext)
        a.body_prompt = "short"
        bm = fb.load(spec["body"])
        mlines, _names = dr.move_menu(a, bm)
        serve = pz.system_prompt(bm, a.persona, False, "", mlines,
                                 a.body_prompt)
        assert s["messages"][0]["content"] == serve, name
        assert len(serve) < 0.5 * len(t["messages"][0]["content"])
    assert {"seed-san"} <= seen


def test_export_keeps_the_gesture_entry_of_the_run(beats2_run):
    out, recs = beats2_run
    rec = next(r for r in recs if r["avatar"]["name"] == "Hiyori")
    bm = fb.load(RIGS[0])
    gl = dr.gesture_lines({"wave_a": "wave_a", "nod_b": "nod_b"},
                          {"wave_a": "a big wave", "nod_b": ""})
    ml = _menu(bm)
    persona = "Hiyori, a girl."
    teacher = pz.system_prompt(bm, persona, False, gl, ml, "beats2", 3)
    assert pz.teacher_parts(teacher) == (persona, gl, True)
    r = {"avatar": {"name": "Hiyori", "body": rec["avatar"]["body"],
                    "variant": 3}}
    got = dx.student_system(teacher, r, "short", bm)
    assert got == pz.system_prompt(bm, persona, False, gl, ml, "short")
    assert gl in got
    # no gestures, no moves
    t0 = pz.system_prompt(bm, persona, False, "", "", "beats2")
    assert pz.teacher_parts(t0) == (persona, "", False)
    r["avatar"]["variant"] = 0
    assert dx.student_system(t0, r, "short", bm) \
        == pz.system_prompt(bm, persona, False, "", "", "short")


def test_export_applies_to_beats2_data_only(tmp_path):
    out, recs = _run(tmp_path, FACES, avatars="seed-san", sessions=2,
                     turns=2, prompt="beats")
    with pytest.raises(SystemExit, match="beats2"):
        _examples(out, recs, "short")
    bm = fb.load(RIGS[0])
    t = pz.system_prompt(bm, "X.", False, "", "", "beats2")
    rec = {"avatar": {"name": "Hiyori", "body": RIGS[0], "variant": 0}}
    # a teacher prompt that does not come from this body file is refused
    with pytest.raises(SystemExit, match="byte for byte"):
        dx.student_system(t.replace("one to three", "one to four"), rec,
                          "short", bm)
    with pytest.raises(SystemExit):
        dx.student_system(t, rec, "teacher", bm)


def test_beats_defaults_still_detect_the_data(beats2_run):
    out, _recs = beats2_run
    assert dx.default_max_beats(out) == dx.BEATS_MAX
    assert dx.default_max_hands(out) == dx.MAX_HANDS
    assert dx.default_lang_filter(out) is True


def test_cli_flags_and_the_data_hash(beats2_run, tmp_path, capsys):
    from flow.train import sft
    out, _recs = beats2_run
    # flow.dataexport sft --student-prompt short
    dst = tmp_path / "short.jsonl"
    assert dx.main(["sft", out, "--out", str(dst), "--keep-lite",
                    "--student-prompt", "short"]) == 0
    first = json.loads(dst.read_text().splitlines()[0])
    assert first["messages"][0]["content"].count("Examples of placement") \
        == 0
    base = tmp_path / "teacher.jsonl"
    dx.main(["sft", out, "--out", str(base), "--keep-lite"])
    assert "Examples of placement" in json.loads(
        base.read_text().splitlines()[0])["messages"][0]["content"]
    # flow.train.sft: the examples, and the data hash
    on = sft.load_examples(out, "session", keep_lite=True,
                           student_prompt="short")
    off = sft.load_examples(out, "session", keep_lite=True)
    assert on[0]["messages"][0]["content"] == first["messages"][0]["content"]
    assert on[0]["messages"][1:] == off[0]["messages"][1:]
    rows = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}]
    assert sft.data_hash(rows) == sft.data_hash(rows, "teacher")
    assert sft.data_hash(rows) != sft.data_hash(rows, "short")
    p = sft.argparse.ArgumentParser()
    sft.add_data_args(p)
    assert p.parse_args(["--data", out]).student_prompt == "teacher"
    assert p.parse_args(["--data", out, "--student-prompt",
                         "short"]).student_prompt == "short"
    # an sft.jsonl holds its prompts already
    with pytest.raises(SystemExit, match="dataset folder"):
        sft.load_examples(str(base), student_prompt="short")


def test_evalsft_flag_rebuilds_the_same_prompt(beats2_run):
    from flow.train import evalsft
    out, recs = beats2_run
    o = evalsft.parse(["run", "--data", out, "--split", "s.json", "--out",
                       "o", "--name", "student", "--student-prompt",
                       "short"])
    assert o.student_prompt == "short"
    assert evalsft.parse(["run", "--data", out, "--split", "s.json",
                          "--out", "o", "--name", "x"]).student_prompt \
        == "teacher"
    # the prompt it sends is the export's
    reg = dx.load_registry(out)
    r = recs[0]
    teacher = reg["prompts"][r["avatar"]["prompt_sha"]]
    ex = dx.student_system(teacher, r, "short")
    assert len(ex) < 0.5 * len(teacher)
    sh = open(os.path.join(ROOT, "scripts", "eval_sft.sh")).read()
    assert "STUDENT_PROMPT" in sh and "--student-prompt" in sh
