"""flow.live: the first-token / stall watchdog, the backend warm-up and the
batching of chat lines that arrive while she talks. Fake backends; short
timeouts."""

from __future__ import annotations

import io
import json
import threading
import time

import pytest

from flow import director as dr
from flow import live as lv
from tests.test_live import _events, _performer


class Hang:
    """Never yields (until released); counts the streams it was asked for."""

    stop_reason = None

    def __init__(self):
        self.calls, self.release, self.cancelled = 0, threading.Event(), 0

    def stream(self, system, messages):
        self.calls += 1
        self.release.wait(10)
        yield "LATE TOKENS THAT MUST NOT BE SPOKEN. "

    def cancel(self):
        self.cancelled += 1


class HangThenSay(Hang):
    """The first stream hangs, the next ones answer."""

    def stream(self, system, messages):
        self.calls += 1
        if self.calls == 1:
            self.release.wait(10)
            yield "OLD ATTEMPT. "
            return
        yield "Second try works. "


class Slow:
    """Yields after a delay shorter than the timeout."""

    stop_reason = None

    def stream(self, system, messages):
        time.sleep(0.15)
        yield "Slow but fine. "
        time.sleep(0.15)
        yield "Still here."


class StallMid(Hang):
    def stream(self, system, messages):
        self.calls += 1
        yield "First part. Second par"
        self.release.wait(10)
        yield "t that must not appear."


def _perf(tmp_path, backend, first=0.4, stall=0.4):
    p, hub, stream, clock, tts = _performer(tmp_path, "")
    p.backend = backend
    p.first_timeout, p.stall_timeout = first, stall
    logs = []
    p.log = logs.append
    return p, hub, logs


def test_a_backend_that_never_yields_is_retried_then_the_turn_dropped(tmp_path):
    be = Hang()
    p, hub, logs = _perf(tmp_path, be)
    t0 = time.monotonic()
    p.turn("hi")
    assert time.monotonic() - t0 < 5
    assert be.calls == 2 and be.cancelled == 2
    assert any("retrying" in m for m in logs)
    assert any("dropping this turn" in m for m in logs)
    # History stays consistent: the user line is popped, nothing added.
    assert p.messages == []
    assert _events(hub, "audio") == []
    assert _events(hub, "reply")[0]["stop_reason"] == "stalled"
    # The loop goes on: a later turn on a working backend is answered.
    be.release.set()
    p.backend = Slow()
    p.turn("again")
    assert [m["role"] for m in p.messages] == ["user", "assistant"]
    assert p.messages[-1]["content"] == "Slow but fine. Still here."


def test_the_retry_answers_and_the_old_attempt_cannot_leak(tmp_path):
    be = HangThenSay()
    p, hub, logs = _perf(tmp_path, be)
    p.turn("hi")
    assert any("retrying" in m for m in logs)
    assert p.messages[-1] == {"role": "assistant",
                              "content": "Second try works. "}
    be.release.set()                   # the abandoned thread wakes up now
    time.sleep(0.2)
    p.backend = Slow()
    p.turn("next")
    assert "OLD ATTEMPT" not in p.messages[-1]["content"]
    assert "OLD ATTEMPT" not in "".join(m["text"] for m in
                                        _events(hub, "audio"))


def test_a_slow_start_under_the_timeout_is_not_aborted(tmp_path):
    p, hub, logs = _perf(tmp_path, Slow(), first=2.0, stall=2.0)
    p.turn("hi")
    assert not any("retrying" in m or "dropping" in m for m in logs)
    assert p.messages[-1]["content"] == "Slow but fine. Still here."


def test_a_stall_mid_stream_keeps_what_was_said(tmp_path):
    be = StallMid()
    p, hub, logs = _perf(tmp_path, be, first=2.0, stall=0.3)
    p.turn("hi")
    assert be.calls == 1               # no retry once words were spoken
    assert any("mid-reply" in m for m in logs)
    said = [m["text"] for m in _events(hub, "audio")]
    assert said[0] == "First part."
    assert "Second par" in "".join(said)       # the held words are spoken
    assert p.messages[-1]["role"] == "assistant"
    assert p.messages[-1]["content"] == "First part. Second par"
    be.release.set()
    time.sleep(0.1)
    assert p.messages[-1]["content"] == "First part. Second par"


def test_zero_disables_the_watchdog(tmp_path):
    p, hub, logs = _perf(tmp_path, Slow(), first=0, stall=0)
    p.turn("hi")
    assert p.messages[-1]["content"] == "Slow but fine. Still here."


# ------------------------------------------------------------ the backend

class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_ollama_request_has_keep_alive_and_timeout_and_cancel(monkeypatch):
    sent = []
    resp = FakeResp(b'{"message":{"content":"hi"}}\n'
                    b'{"done":true,"done_reason":"stop"}\n')
    be = dr.OllamaBackend("m", timeout=7, keep_alive="30m", log=lambda s: None)
    monkeypatch.setattr(be, "_post", lambda body, to: (sent.append((body, to)),
                                                       resp)[1])
    assert "".join(be.stream("sys", [{"role": "user", "content": "x"}])) == "hi"
    body, to = sent[0]
    assert body["keep_alive"] == "30m" and to == 7
    assert be.stop_reason == "stop" and not be._open
    # Defaults of other callers: no keep_alive in the body, 300 s.
    d = dr.OllamaBackend("m", log=lambda s: None)
    assert d.timeout == 300 and d.keep_alive is None
    assert dr.OllamaBackend("m", keep_alive="300").keep_alive == 300


def test_warmup_loads_with_no_messages(monkeypatch):
    sent = []
    be = dr.OllamaBackend("m", keep_alive="30m")
    monkeypatch.setattr(be, "_post", lambda body, to: (sent.append(body),
                                                       FakeResp(b"{}"))[1])
    assert be.warmup() >= 0
    assert sent[0]["messages"] == [] and sent[0]["keep_alive"] == "30m"


def test_warm_backend_logs_time_and_only_warns_on_failure():
    class A:
        no_warmup = False

    class Ok:
        def warmup(self):
            return 2.5

    class Bad:
        def warmup(self):
            raise OSError("refused")

    logs = []
    assert lv.warm_backend(A(), Ok(), logs.append) == 2.5
    assert "warmed up in 2.5s" in logs[-1]
    assert lv.warm_backend(A(), Bad(), logs.append) is None
    assert "warm-up failed" in logs[-1]
    A.no_warmup = True
    n = len(logs)
    assert lv.warm_backend(A(), Ok(), logs.append) is None and len(logs) == n
    assert lv.warm_backend(A(), object(), logs.append) is None   # no warmup()


def test_cli_flags_and_live_defaults():
    import argparse
    ap = argparse.ArgumentParser()
    dr.add_args(ap)
    assert ap.parse_args([]).ollama_timeout == 300.0     # others unchanged
    assert ap.parse_args([]).ollama_keep_alive is None


# --------------------------------------------------------------- batching

def test_chat_lines_that_queued_up_are_one_turn(tmp_path):
    p, hub, logs = _perf(tmp_path, Slow())
    for t in ("A", "B", "C"):
        p.say(t)
    text, quiet = p.next_turn()
    assert (text, quiet) == ("A\nB\nC", False)
    assert p.batched == 3
    p.turn(text)
    assert p.messages[0]["content"] == "A\nB\nC"
    assert _events(hub, "user")[0]["batched"] == 3
    assert _events(hub, "reply")[0]["batched"] == 3
    # One line: unchanged, no batch fields.
    p.say("solo")
    assert p.next_turn() == ("solo", False) and p.batched == 1


def test_a_batch_is_capped_and_the_drop_logged(tmp_path):
    p, hub, logs = _perf(tmp_path, Slow())
    for i in range(8):
        p.say(f"m{i}")
    text, _ = p.next_turn()
    assert text.split("\n") == ["m3", "m4", "m5", "m6", "m7"]
    assert any("3 older dropped" in m for m in logs)


def test_a_real_message_wins_over_quiet_and_stop_is_kept(tmp_path):
    p, hub, logs = _perf(tmp_path, Slow())
    p.quiet_after, p.quiet_t0 = 0.01, -100.0
    assert p.quiet_due()
    p.say("hello")
    p.say("there")
    assert not p.quiet_due()
    assert p.next_turn(poll=0.01) == ("hello\nthere", False)
    p.inbox.put("last")
    p.inbox.put(None)
    assert p.next_turn() == ("last", False)
    assert p.next_turn() == (None, False)
    # Nothing queued and quiet is due: the quiet line, batch of one.
    p.quiet_t0 = -100.0
    text, quiet = p.next_turn(poll=0.01)
    assert quiet and p.batched == 1


# ---------------------------------------------------------- context budget

class Usage:
    """A backend that reports prompt_eval_count like Ollama and records
    the messages it was sent."""

    stop_reason = "stop"

    def __init__(self, tokens, reply="ok. "):
        self.tokens, self.reply, self.sent, self.usage = tokens, reply, [], {}

    def stream(self, system, messages):
        self.sent.append(list(messages))
        self.usage = {"prompt_eval_count": self.tokens, "eval_count": 5}
        yield self.reply


def _ctx_perf(tmp_path, backend, ctx=2000):
    tmp_path.mkdir(exist_ok=True)
    p, hub, logs = _perf(tmp_path, backend, first=5, stall=5)
    p.a.ollama_ctx = ctx
    p.a.max_tokens = None
    p.history_turns = 50
    return p, logs


def test_old_exchanges_are_dropped_to_fit_the_budget(tmp_path):
    be = Usage(100)
    p, logs = _ctx_perf(tmp_path, be, ctx=2000)    # budget 2000-1024-256=720
    p.run = {**p.run, "system": "S" * 100}
    for i in range(6):
        p.messages += [{"role": "user", "content": f"u{i} " + "x" * 300},
                       {"role": "assistant", "content": "a" * 300}]
    p.turn("now")
    sent = be.sent[0]
    assert sent[-1]["content"] == "now" and sent[0]["role"] == "user"
    assert len(sent) < 13                       # exchanges were dropped
    assert p.sent_chars(sent) / 2.0 <= 720      # fits (guess ratio, 2 c/t)
    assert any("context budget" in m and "of 720 tokens" in m for m in logs)
    # never the current user turn, even when it alone is over the budget
    p.messages = []
    p.turn("y" * 5000)
    assert be.sent[-1][-1]["content"] == "y" * 5000 and len(be.sent[-1]) == 1


def test_the_reported_count_calibrates_the_ratio_and_warns(tmp_path):
    be = Usage(1900)                            # 95% of a 2000 window
    p, logs = _ctx_perf(tmp_path, be, ctx=2000)
    p.turn("hello there")
    assert p.cpt is not None and p.cpt == pytest.approx(
        p.sent_chars(be.sent[0]) / 1900, rel=0.01) or p.cpt == 1.0
    assert any("WARNING" in m and "1900" in m for m in logs)
    be.tokens = 100                             # the ratio never goes up
    old = p.cpt
    p.turn("again")
    assert p.cpt <= old


def test_max_tokens_below_the_reserve_and_window_sources(tmp_path):
    p, logs = _ctx_perf(tmp_path, Usage(10), ctx=4000)
    assert p.prompt_budget() == 4000 - 1024 - 256
    p.a.max_tokens = 300
    assert p.prompt_budget() == 4000 - 300 - 256
    p2, _ = _ctx_perf(tmp_path / 'b', Usage(10), ctx=None)
    assert p2.context_window() == lv.DEFAULT_CTX
    p3, _ = _ctx_perf(tmp_path / 'c', Usage(10), ctx=None)
    p3.backend.ctx_window = lambda: 16384
    assert p3.context_window() == 16384


def test_ctx_window_reads_num_ctx_from_api_show(monkeypatch):
    import urllib.request

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = json.dumps({"parameters": "stop <x>\nnum_ctx                        8192"})
    monkeypatch.setattr(urllib.request.OpenerDirector, "open",
                        lambda self, req, timeout=None: R(body.encode()))
    assert dr.OllamaBackend("m").ctx_window() == 8192


class BrokenTTS:
    """A TTS whose every synth fails (edge-tts behind a dead proxy)."""

    whole = False

    def __init__(self):
        self.calls = 0

    def begin(self, spoken, lang=None):
        pass

    def synth(self, text, mood=None):
        self.calls += 1
        raise RuntimeError("edge-tts sent nothing complete in 20 s")


def test_a_failing_tts_is_retried_then_the_sentence_goes_silent(tmp_path):
    p, hub, logs = _perf(tmp_path, Slow(), first=2.0, stall=2.0)
    p.tts = BrokenTTS()
    p.turn("hi")
    assert p.tts.calls >= 4            # two sentences, two tries each
    assert any("tts failed" in m and "retrying" in m for m in logs)
    assert any("goes without a voice" in m for m in logs)
    # the turn still ends and is kept
    assert p.messages[-1]["content"] == "Slow but fine. Still here."


def test_edge_tts_is_bounded_and_does_not_use_stream_sync(monkeypatch):
    import asyncio
    import sys
    import types

    from flow import speech as sp

    class Comm:
        def __init__(self, *a, **k):
            pass

        def stream_sync(self):          # the call that froze flow.live
            raise AssertionError("stream_sync must not be used")

        async def stream(self):
            await asyncio.sleep(5)
            yield {"type": "audio", "data": b""}

    monkeypatch.setitem(sys.modules, "edge_tts",
                        types.SimpleNamespace(Communicate=Comm))
    monkeypatch.setattr(sp, "EDGE_TIMEOUT", 0.2)
    t = sp.EdgeTTS(voice="en-US-AvaNeural")
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="nothing complete"):
        t.synth("hello")
    assert time.monotonic() - t0 < 2
