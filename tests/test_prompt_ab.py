"""The prompt A/B of flow.train.evalsft (--prompt-cond full | full+rem |
short+rem): what each condition sends, with fake backends; one student file
per condition; scripts/checks.py comparing them with distinguishable
labels; the shell script's knob. No model."""

from __future__ import annotations

import json
import os

import pytest

from flow import checks as ck
from flow import dataexport as dx
from flow import director as dr
from flow import pose as pz
from flow import rules as rl
from flow.train import evalsft
from tests.test_dataset import ROOT
from tests.test_short_prompt import beats2_run  # noqa: F401 - the fixture

REPLY = ('Hi there, it is so good to see you today! [pose {"head.yaw": -15}]'
         ' Tell me more.')


@pytest.fixture(scope="module")
def split(beats2_run, tmp_path_factory):  # noqa: F811
    out, recs = beats2_run
    sessions = sorted({r["session"] for r in recs})
    path = tmp_path_factory.mktemp("abs") / "split.json"
    path.write_text(json.dumps({"eval_sessions": sessions[:3],
                                "train_sessions": sessions[3:]}))
    return str(path)


class Fake:
    """A backend that records what it is sent."""

    model, stop_reason, usage = "fake", "stop", {}

    def __init__(self, calls, tag):
        self.calls, self.tag = calls, tag

    def stream(self, system, messages):
        self.calls.append({"cond": self.tag, "system": system,
                           "messages": [dict(m) for m in messages]})
        yield REPLY


@pytest.fixture(scope="module")
def runs(beats2_run, split, tmp_path_factory):  # noqa: F811
    """The three conditions on the same held-out sessions."""
    out, recs = beats2_run
    ev = tmp_path_factory.mktemp("abev") / "eval"
    calls, sums = [], {}
    real = dr.make_backend
    try:
        for cond in evalsft.PROMPT_CONDS:
            dr.make_backend = lambda a, log=print, c=cond: Fake(calls, c)
            argv = ["run", "--data", out, "--split", split, "--out",
                    str(ev), "--name", "student"]
            if cond != "full":
                argv += ["--prompt-cond", cond]
            sums[cond] = evalsft.run(evalsft.parse(argv), log=lambda s: None)
    finally:
        dr.make_backend = real
    return out, recs, ev, calls, sums


def by_cond(calls, cond):
    return [c for c in calls if c["cond"] == cond]


def rows(ev, name):
    with open(ev / f"{name}.jsonl", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


# --------------------------------------------------------------- the files

def test_each_condition_writes_its_own_files(runs):
    _out, _recs, ev, _calls, sums = runs
    assert sorted(os.listdir(ev)) == [
        "student.full+rem.jsonl", "student.full+rem.summary.json",
        "student.jsonl", "student.short+rem.jsonl",
        "student.short+rem.summary.json", "student.summary.json"]
    assert sums["full"]["name"] == "student"
    assert sums["full+rem"]["name"] == "student.full+rem"
    assert sums["short+rem"]["cond"] == "short+rem"
    for name, cond in (("student", "full"), ("student.full+rem", "full+rem"),
                       ("student.short+rem", "short+rem")):
        rs = rows(ev, name)
        assert len(rs) == 9 and {r["cond"] for r in rs} == {cond}
        assert {r["name"] for r in rs} == {name}
        assert [r["id"] for r in rs] == [r["id"] for r in rows(ev, "student")]
    assert not any("rules" in r for r in rows(ev, "student"))
    text = evalsft.compare(str(ev))
    for name in ("student", "student.full+rem", "student.short+rem"):
        assert name in text


# ------------------------------------------------------ what is sent

def test_full_sends_what_it_always_did(runs):
    out, recs, _ev, calls, _s = runs
    reg = dx.load_registry(out)
    full = by_cond(calls, "full")
    assert len(full) == 9
    teacher = {r["session"]: reg["prompts"][r["avatar"]["prompt_sha"]]
               for r in recs}
    for c in full:
        assert c["system"] in teacher.values()
        assert all("(reminders" not in m["content"] for m in c["messages"])
        assert c["messages"][-1]["role"] == "user"


def test_full_plus_rem_appends_the_reminders_to_the_last_user_turn(runs):
    _out, _recs, ev, calls, _s = runs
    full, rem = by_cond(calls, "full"), by_cond(calls, "full+rem")
    assert len(rem) == 9
    asked_wave = 0
    for a, b, row in zip(full, rem, rows(ev, "student.full+rem")):
        assert b["system"] == a["system"]              # the data's prompt
        # history: the same bytes, with the model's own replies
        assert b["messages"][:-1] == a["messages"][:-1]
        assert all("(reminders" not in m["content"] for m in b["messages"][:-1])
        prev = next((m["content"] for m in reversed(b["messages"][:-1])
                     if m["role"] == "assistant"), "")
        from flow import body as fb
        can = rl.capabilities(fb.load(f"rigs/{row['avatar']}.body.json"))
        block, ids = rl.block(row["user"], prev, can=can)
        assert row["rules"] == ids
        last = b["messages"][-1]["content"]
        if block:
            # exactly as flow.live puts it: the last line of the message
            assert last == a["messages"][-1]["content"] + "\n" + block
            assert last.split("\n")[-1].startswith("(reminders: ")
        else:
            assert last == a["messages"][-1]["content"]
        asked_wave += "wave-move" in ids
    assert asked_wave >= 1
    # the Japanese line after an English reply: a language switch
    assert any("language" in r["rules"] for r in rows(ev, "student.full+rem"))


def test_short_plus_rem_sends_the_short_prompt_the_director_serves(runs,
                                                                   split):
    out, recs, _ev, calls, _s = runs
    held = set(json.load(open(split))["eval_sessions"])
    reg = dx.load_registry(out)
    rem, short = by_cond(calls, "full+rem"), by_cond(calls, "short+rem")
    full = by_cond(calls, "full")
    assert len(short) == 9
    cfg_recs = {r["session"]: r for r in recs if r["session"] in held}
    seen = 0
    for a, b, c in zip(full, rem, short):
        # the user turns and reminders are the full+rem run's, the system
        # prompt is the short one
        assert c["messages"] == b["messages"]
        assert c["system"] != a["system"] and len(c["system"]) < \
            0.5 * len(a["system"])
        assert c["system"].startswith("You are ")
        seen += 1
    assert seen == 9
    # bytes: the export's rebuild, and the director's own construction
    # (flow.pose.system_prompt "short" from the body file, the persona and
    # the gesture / move entries parsed out of the teacher prompt)
    for r in cfg_recs.values():
        teacher = reg["prompts"][r["avatar"]["prompt_sha"]]
        ex = dx.student_system(teacher, r, "short")
        bm = dx.ds.fb.load(dx._body_for(r))
        persona, glines, moves_on = pz.teacher_parts(teacher)
        mlines = dx.ds.mv.prompt_lines(bm) if moves_on else ""
        served = pz.system_prompt(bm, persona, False, glines, mlines, "short")
        assert ex == served
        assert any(c["system"] == served for c in short)
    # every request of the condition carries one of those, byte for byte
    sent = {c["system"] for c in short}
    assert len(sent) == len({r["avatar"]["name"] for r in cfg_recs.values()})


def test_with_reminders_leaves_the_history_alone():
    hist = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello there!"}]
    msgs = hist + [{"role": "user", "content": "Look to your right\n(x)"}]
    snap = json.loads(json.dumps(msgs))
    sent, ids = evalsft.with_reminders(msgs, "Look to your right", hist)
    assert msgs == snap and ids == ["look-side", "her-right"]
    assert sent[:-1] == hist
    assert sent[-1]["content"] == (
        "Look to your right\n(x)\n" + rl.block("Look to your right")[0])
    sent, ids = evalsft.with_reminders(msgs, "How are you?", hist)
    assert ids == [] and sent == msgs


def test_conditions_that_do_not_make_sense_are_refused(beats2_run, split):  # noqa: F811
    out, _recs = beats2_run
    base = ["run", "--data", out, "--split", split, "--out", "o", "--name",
            "student"]
    with pytest.raises(SystemExit, match="short\\+rem"):
        evalsft.run(evalsft.parse(base + ["--prompt-cond", "full+rem",
                                          "--student-prompt", "short"]))
    with pytest.raises(SystemExit, match="recorded"):
        evalsft.run(evalsft.parse(base + ["--prompt-cond", "short+rem",
                                          "--recorded"]))
    assert evalsft.parse(base).prompt_cond == "full"
    with pytest.raises(SystemExit):
        evalsft.parse(base + ["--prompt-cond", "short"])


# ---------------------------------------------------- scripts/checks.py

def test_checks_compares_the_three_side_by_side(runs, tmp_path, capsys):
    _out, _recs, ev, _calls, _s = runs
    # the folder: three inputs, labelled by condition
    got = ck.load(str(ev))
    labels = [lab for lab, _r in got]
    assert labels == ["eval:student", "eval:student.full+rem",
                      "eval:student.short+rem"]
    assert all(len(r) == 9 for _l, r in got)
    # the files
    files = [str(ev / f"{n}.jsonl") for n in
             ("student", "student.full+rem", "student.short+rem")]
    labels = [ck.load(f)[0][0] for f in files]
    assert labels == ["eval/student", "eval/student.full+rem",
                      "eval/student.short+rem"]
    # the script prints one column per condition
    assert ck.main(files + ["--json", str(tmp_path / "c.json")]) == 0
    text = capsys.readouterr().out
    head = text.splitlines()[0]
    for lab in labels:
        assert lab in head
    assert list(json.load(open(tmp_path / "c.json"))) == labels
    # the same condition of two runs keeps two columns
    other = tmp_path / "run2" / "eval"
    other.mkdir(parents=True)
    (other / "student.jsonl").write_text(open(files[0]).read())
    assert ck.main([files[0], str(other / "student.jsonl")]) == 0
    head = capsys.readouterr().out.splitlines()[0]
    assert "eval/student" in head and str(other / "student") in head


def test_the_script_runs_the_conditions():
    sh = open(os.path.join(ROOT, "scripts", "eval_sft.sh")).read()
    assert "PROMPT_CONDS" in sh and "--prompt-cond" in sh
    assert "student.$cond" in sh or 'student.${cond}' in sh
    import subprocess
    assert subprocess.run(["bash", "-n", os.path.join(
        ROOT, "scripts", "eval_sft.sh")]).returncode == 0
