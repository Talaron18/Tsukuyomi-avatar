"""
flow.train.evalsft without a model: the held-out sessions of a split
replayed turn by turn through a backend (the recorded teacher replies, a
mocked student that keeps its own history), the per-turn numbers (parse,
refusals by text, markers, unplayed, reach, latency) and their summary,
the comparison table and the A/B pack's director runs (not rendered).
Pure numpy (the generation is test_dataset's, mocked).
"""

from __future__ import annotations

import json
import os

import pytest

from flow import dataset as ds
from flow import director as dr
from flow.train import evalsft
from tests.test_dataset import REPLY, CharLLM, UserLLM, _opts


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("ev") / "run"
    res = ds.generate(_opts(out, sessions=4, turns=2),
                      backend_factory=lambda role, av: CharLLM() if
                      role == "character" else UserLLM(), log=lambda s: None)
    assert res["ok"] == 4
    sessions = sorted({r["session"] for r in ds.iter_records(str(out))})
    split = out / "split.json"
    split.write_text(json.dumps({"eval_sessions": sessions[:3],
                                 "train_sessions": sessions[3:]}))
    return out, split


def test_refusal_text():
    assert evalsft.refusal_text("I'm sorry, but I can't do that.")
    assert evalsft.refusal_text("As an AI, I have no body.")
    assert evalsft.refusal_text("抱歉，我不能")
    assert evalsft.refusal_text("")
    assert not evalsft.refusal_text("Sorry I'm late! [pose {}]")


def test_recorded_teacher_and_a_student(data, tmp_path, monkeypatch):
    out, split = data
    ev = tmp_path / "eval"
    o = evalsft.parse(["run", "--data", str(out), "--split", str(split),
                       "--out", str(ev), "--name", "teacher", "--recorded"])
    s = evalsft.run(o, log=lambda s: None)
    assert s["sessions"] == 3 and s["turns"] == 6
    assert s["parse_rate"] == 0 and s["refusal_text_rate"] == 0
    assert s["markers_per_turn"]["pose"] == 2 and s["unplayed"] == {"nod": 6}
    assert s["reach"]["targets"] == 18 and s["first_text_seconds"]["p50"] \
        is not None
    rows = [json.loads(x) for x in open(ev / "teacher.jsonl")]
    assert [r["reply_raw"] for r in rows] == [REPLY] * 6
    # A student: its own replies are its history.
    sent = []

    class Student(CharLLM):
        def stream(self, system, messages):
            sent.append(messages)
            yield from super().stream(system, messages)
    bad = "I'm sorry, but I can't. [pose {\"nope.x\": 1}]"
    monkeypatch.setattr(dr, "make_backend",
                        lambda a, log=print: Student([bad, REPLY]))
    o = evalsft.parse(["run", "--data", str(out), "--split", str(split),
                       "--out", str(ev), "--name", "student"])
    s2 = evalsft.run(o, log=lambda s: None)
    assert s2["refusal_text_rate"] == 0.5 and s2["parse_rate"] == 0.5
    assert sent[1][1] == {"role": "assistant", "content": bad}
    rows = [json.loads(x) for x in open(ev / "student.jsonl")]
    assert rows[0]["tokens_per_second"] is None or \
        rows[0]["tokens_per_second"] > 0
    text = evalsft.compare(str(ev))
    assert "teacher" in text and "student" in text
    # The A/B pack: each model's first turns played by the director
    # (lite bodies have no model to render: skipped).
    o = evalsft.parse(["ab", "--data", str(out), "--out", str(ev),
                       "--no-render", "--n", "1", "--tts", "none"])
    evalsft.ab(o, log=lambda s: None)
    vrm = [d for d in (ev / "ab").glob("*/teacher/0")]
    if vrm:
        assert (vrm[0] / "actions.json").exists()
        assert (vrm[0].parent.parent / "student" / "0" / "reply.txt").exists()
    assert os.path.isdir(ev / "ab")


# ------------------------------------------------- vllm backend and --conc

def _sse(events):
    import io
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return io.BytesIO((body + "data: [DONE]\n\n").encode())


def test_vllm_backend_body(monkeypatch):
    import urllib.request
    seen = {}

    def urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _sse([{"choices": [{"delta": {"content": "Hi"},
                                   "finish_reason": "stop"}]}])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    o = evalsft.parse(["run", "--data", "d", "--split", "s", "--out", "o",
                       "--name", "student", "--backend", "vllm",
                       "--model", "q8r3"])
    be = dr.make_backend(o, log=lambda s: None)
    r = dr.ask(be, "SYS", [{"role": "user", "content": "q"}], [],
               log=lambda s: None)
    assert r["spoken"] == "Hi"
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer EMPTY"
    b = seen["body"]
    assert b["model"] == "q8r3" and b["stream"] is True
    assert (b["temperature"], b["top_p"], b["top_k"]) == (0.7, 0.8, 20)
    assert "repetition_penalty" not in b and "enable_thinking" not in b
    assert b["messages"][0] == {"role": "system", "content": "SYS"}
    # --api-base, the key variable and the family's sampling
    monkeypatch.setenv("VLLM_API_KEY", "k")
    o = evalsft.parse(["run", "--data", "d", "--split", "s", "--out", "o",
                       "--name", "student", "--backend", "vllm", "--model",
                       "m", "--api-base", "http://h:9/v1/",
                       "--vllm-family", "gemma3"])
    list(dr.make_backend(o, log=lambda s: None).stream("S", []))
    assert seen["url"] == "http://h:9/v1/chat/completions"
    assert seen["auth"] == "Bearer k" and seen["body"]["top_k"] == 64
    # a repeat penalty maps to vLLM's name
    monkeypatch.setitem(chat_sampling(), "qwen3",
                        {"temperature": 0.5, "top_p": 1, "top_k": 5,
                         "repeat_penalty": 1.1})
    o.vllm_family = "qwen3"
    list(dr.make_backend(o, log=lambda s: None).stream("S", []))
    assert seen["body"]["repetition_penalty"] == 1.1
    assert "repeat_penalty" not in seen["body"]


def chat_sampling():
    from flow.train import chat
    return chat.SAMPLING


def test_other_backends_send_no_sampling(monkeypatch):
    import urllib.request
    seen = {}

    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _sse([])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    list(dr.OpenAICompatBackend("deepseek", "m").stream("S", []))
    assert set(seen["body"]) == {"model", "stream", "max_tokens",
                                 "stream_options", "messages"}
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(SystemExit):
        list(dr.OpenAICompatBackend("deepseek", "m").stream("S", []))


def test_conc_keeps_the_serial_run(data, tmp_path, monkeypatch):
    import threading
    import time
    out, split = data
    lock = threading.Lock()
    live, peak, waiting = [0], [0], [False]

    class Fake(CharLLM):
        def stream(self, system, messages):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            # In the concurrent run, stay in the stream until another
            # session's is open too (up to 2 s): on a slow machine a fixed
            # short sleep let each session finish before the next began.
            t0 = time.monotonic()
            while (waiting[0] and peak[0] < 2
                   and time.monotonic() - t0 < 2.0):
                time.sleep(0.01)
            time.sleep(0.05)
            try:
                # a reply that depends on the history, so a mixed-up
                # session would show
                self.replies = [REPLY + f" {len(messages)}"]
                self.k = 0
                yield from super().stream(system, messages)
            finally:
                with lock:
                    live[0] -= 1

    monkeypatch.setattr(dr, "make_backend",
                        lambda a, log=print: Fake([REPLY]))
    res = {}
    for conc in (1, 4):
        ev = tmp_path / f"c{conc}"
        argv = ["run", "--data", str(out), "--split", str(split),
                "--out", str(ev), "--name", "student"]
        waiting[0] = conc > 1
        o = evalsft.parse(argv + (["--conc", str(conc)] if conc > 1 else []))
        logs = []
        evalsft.run(o, log=logs.append)
        res[conc] = [json.loads(x) for x in open(ev / "student.jsonl")]
        if conc > 1:
            assert all(x.startswith("[") for x in logs[:-1] if "evalsft" in x)
    assert peak[0] > 1
    key = lambda rows: [(r["id"], r["session"], r["turn"], r["reply_raw"],
                         r["markers"], r["reach"]["ok"], r["reach"]["targets"],
                         r["parse"]) for r in rows]
    assert key(res[1]) == key(res[4]) and len(res[1]) == 6
    assert [r["session"] for r in res[4]] == sorted(r["session"]
                                                     for r in res[4])
    assert evalsft.parse(["run", "--data", "d", "--split", "s", "--out", "o",
                          "--name", "x"]).conc == 1


def test_eval_sft_sh_passes_conc_and_vllm(tmp_path):
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.txt"
    shim = bindir / "python"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ $1 == -c ]]; then exec {sys.executable} "$@"; fi\n'
        f'echo "$@" >> {log}\n')
    shim.chmod(0o755)
    run = tmp_path / "run"
    run.mkdir()
    (run / "train.json").write_text('{"family": "qwen3"}')
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
               RUN=str(run), DATA=str(tmp_path), TEACHER="1",
               TEACHER_BACKEND="deepseek", STUDENT_BACKEND="vllm",
               STUDENT_MODEL="q8r3", CONC="4",
               VLLM_URL="http://h:8000/v1")
    r = subprocess.run(["bash", os.path.join(root, "scripts",
                                             "eval_sft.sh")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    st = next(c for c in calls if "--name student" in c)
    assert "--backend vllm" in st and "--model q8r3" in st
    assert "--conc 4" in st and "--api-base http://h:8000/v1" in st
    assert "--vllm-family qwen3" in st
    assert "--conc 4" in next(c for c in calls if "--name teacher" in c)
    assert "--conc" not in next(c for c in calls if "--name recorded" in c)
    # the default is unchanged
    env.pop("CONC")
    env["STUDENT_BACKEND"] = "ollama"
    log.write_text("")
    subprocess.run(["bash", os.path.join(root, "scripts", "eval_sft.sh")],
                   env=env, capture_output=True, text=True)
    assert all("--conc" not in c for c in log.read_text().splitlines())
