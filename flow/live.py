#!/usr/bin/env python3
"""
Live director: talk to the character in a browser and watch it answer, the
idle running on between turns.

flow.director plans a turn offline (reply -> tags -> TTS -> runtime plan ->
render subprocess -> demo.mp4). This runs the same pieces continuously in
one process on the server:

  render     one Live2DBackend on the main thread (GL contexts are thread
             bound), fixed timestep, frame f on screen at f / fps of one
             server clock (time.monotonic from start). flow.stream gives
             each frame's parameters: the idle generated ahead in rolling
             windows, a reply's tokens merged into its future. When the CPU
             cannot keep up, the GL render of late frames is skipped (their
             parameter state is still stepped: switches, blinks and token
             commits are stateful) and the next render advances physics by
             the time that passed (backend.dt), so the motion never drifts
             behind the clock. Frames go to JPEG (Pillow) on an encoder
             thread. The achieved fps is logged every 10 s.
  turn       a line from the page goes to the language model, streamed;
             SentenceCutter cuts its text at sentence ends as it arrives.
             Each sentence, as soon as it is complete: tags parsed
             (director.TagParser), its mood read (speech.mood_of), spoken
             (speech.speak through the VoiceRouter), its tags timed by the
             words / moras, its mouth curve computed (speech.mouth_track),
             then everything put at one server time t: the audio starts at
             t, the mouth curve on frame t * fps, the tags realized
             (director.realize) at t + their offset and merged into the
             stream. t is the later of now + --lead and the end of the
             previous sentence's audio + --gap, snapped to a frame. The
             model streams on while a sentence is spoken; the TTS of the
             next sentence runs while this one plays. Latencies per stage
             are logged and sent to the page.
  viewer     a stdlib HTTP server (127.0.0.1 by default; open it through an
             SSH tunnel). The page reads frames from /video, a
             close-delimited stream of [magic, frame, server time, length,
             JPEG], events from /events (server-sent: audio, subtitles,
             log, reply), audio clips from /audio/<id>.wav, and posts
             lines to /say. /mjpeg is a plain multipart stream for any
             player, with no sync.
  sync       everything carries a server-clock time. The page estimates
             its offset to that clock from /clock pings (NTP style: the
             sample with the smallest round trip, offset = server time -
             midpoint of the ping), then plays both media at server time
             + a fixed playout delay D (?delay=, default 0.3 s): a frame is
             drawn when its time + D comes, an audio clip is started by
             WebAudio at its time + D mapped through getOutputTimestamp()
             (which also takes the output latency out). Audio is pushed
             --lead ahead of its time, so it is always there first; frames
             arrive after render + encode + network, which D must cover
             (the page shows how late frames arrive). Both media on one
             delay keep lip sync whatever the network delay, as long as it
             stays under D.
  record     --record DIR: video.mp4 (every frame on the clock, a skipped
             or dropped frame filled with the next), speech.wav (the clips
             mixed at their times), subs.srt, turns.jsonl (lines, replies,
             tags, latencies) and session.mp4 with all of them muxed,
             written at exit.

  memory     --memory notes|summary (flow.memory): a small JSON state (game,
             corrections, facts; summary: and a rolling summary of the
             trimmed history) kept by a background call after each turn
             and put on the last line of the latest user message. Each
             consolidation is also a {"kind": "memory", ...} line of
             turns.jsonl, and the notes are in memory.json.

  reminders  --reminders auto|off (flow.rules, default off): a small rule
             graph (flow/rules.json): the nodes a turn triggers (look to a
             side, look at me, wave, dance, a place near her, a correction,
             a game, quiet chat, a language switch) with the nodes they need,
             at most 3, as "(reminders: ...)" in the same tail block as the
             memory notes. Logged "rules: turn N: <ids>".

  repair     --repair auto|log|off (flow.repair): each sentence's markers
             are run through the checkers of flow.checks (look side, look
             at me, [frame full] before a dance, a wave that was asked for)
             before it is planned, a checkable failure fixed
             deterministically in the current sentence (never one already
             spoken) and logged as "repair: turn N.k <check>: <before> ->
             <after>"; each is a {"kind": "repair", ...} line of
             turns.jsonl, and a pair (raw reply rejected, repaired chosen)
             goes to repair_pairs.jsonl beside it for a later DPO. The
             history keeps the reply as she performed it, the repaired one.

  body       --mode body (docs/embodiment.md E2, flow.pose): the model
             gets its measured body in the system prompt and, with each
             line, where its body is now (the last frame plus the poses
             still to come) and, with --body-image on a backend that takes
             images, the frame on screen. It writes [pose {...}] markers;
             each sentence's markers are solved with the prior toward that
             pose and merged into the stream as pose tokens beside the
             tags' tokens; the Core check of each solve runs after they are
             on the stream, and its per-target report goes to the log, the
             page and turns.jsonl.

Usage (see --help; the rig, model and TTS arguments are flow.director's):
    xvfb-run -a -s "-screen 0 1024x1024x24" python -m flow.live \\
        --model-json M --rig rigs/R.npz --clips clips2/R.npz --events ev.json \\
        --backend deepseek --tts auto --persona "..." --port 8765 \\
        [--record runs/live/R]
    then on your machine: ssh -L 8765:127.0.0.1:8765 server, and open
    http://127.0.0.1:8765/ (add --mode body [--body-image] for body
    targets instead of tags)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    from . import director as dr
    from . import framing as fr
    from . import memory as mem
    from . import micro as mc
    from . import moves as mv
    from . import pose as pz
    from . import repair as rp
    from . import speech as sp
    from . import stream as fs
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import director as dr
    from flow import framing as fr
    from flow import memory as mem
    from flow import micro as mc
    from flow import moves as mv
    from flow import pose as pz
    from flow import repair as rp
    from flow import speech as sp
    from flow import stream as fs

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.html")
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
VRM_PAGE = os.path.join(WEB, "vrm.html")
FRAME_MAGIC = b"TSKF"
# <magic 4s, frame u32, server time f64, JPEG length u32>, little endian.
FRAME_HEAD = struct.Struct("<4sIdI")


class Clock:
    """The one server clock: seconds since start, monotonic."""

    def __init__(self, t0=None):
        self.t0 = time.monotonic() if t0 is None else t0

    def now(self):
        return time.monotonic() - self.t0


# ------------------------------------------------------------ sentences

LATIN_END = set(".!?")
# Characters of spoken text before a sentence is cut at a clause mark
# anyway: the first sentence sooner, since the first audio waits on it. A
# Latin character is ~1/14 s spoken, a CJK one ~1/6 s (director's rates):
# 60 Latin ~ 4 s, 24 CJK ~ 4 s of speech.
FIRST_CAP = {"latin": 60, "cjk": 24}
CAP = {"latin": 160, "cjk": 64}


class SentenceCutter:
    """Cut a streamed tagged reply into sentences as its chunks arrive.

    feed(chunk) returns the sentences completed by it, close() the rest
    (raw text, tags and all; director.TagParser parses each). A sentence
    ends after a run of sentence ends (".", "...", "?!", "。", "…", a
    closing quote or bracket after them) once a character that is not part
    of the run arrives. A Latin . ! or ? ends one only before whitespace
    (or a tag), so "3.14", "e.g.x" and "Mr.Smith" do not; 。！？… end one
    whatever follows. Nothing inside an open [tag] is looked at, and a tag
    written after a full stop starts the next sentence, where it sits in
    the speech. A sentence running past FIRST_CAP / CAP spoken characters
    is cut at its next clause mark (, 、 ; ...), to bound the wait for the
    first audio.
    """

    def __init__(self, first_cap=None, cap=None, max_tag=None):
        self.first_cap = first_cap or FIRST_CAP
        self.cap = cap or CAP
        # Body mode's pose markers hold JSON (flow.pose.MAX_MARKER).
        self.max_tag = max_tag or dr.MAX_TAG
        self.buf = ""
        self.i = 0                  # next character of buf to look at
        self.count = 0              # spoken characters since the last cut
        self.cjk = 0
        self.n = 0                  # sentences cut so far

    def _limit(self):
        caps = self.first_cap if self.n == 0 else self.cap
        return caps["cjk" if self.cjk * 2 > self.count else "latin"]

    def _cut(self, j, out):
        s = self.buf[:j]
        self.buf, self.i = self.buf[j:], 0
        self.count = self.cjk = 0
        if s.strip():
            out.append(s)
            self.n += 1

    def feed(self, chunk):
        self.buf += chunk.translate(dr._FULLWIDTH)
        out = []
        while self.i < len(self.buf):
            b, i = self.buf, self.i
            c = b[i]
            if c == "[":
                k = b.find("]", i + 1, i + self.max_tag + 2)
                if k >= 0:
                    self.i = k + 1
                    continue
                if len(b) - i <= self.max_tag + 1:
                    break                      # the tag may still close
                # Too long to be a tag: text (the parser keeps it too).
            if c in sp.SENT_END:
                j = i + 1
                while j < len(b) and b[j] in sp.SENT_TAIL:
                    j += 1
                if j >= len(b):
                    self.i = i
                    break                      # the run may go on
                if (c in LATIN_END and all(x in LATIN_END | sp.SENT_TAIL
                                           for x in b[i:j])
                        and not (b[j].isspace() or b[j] == "[")):
                    self.i = j
                    continue
                self._cut(j, out)
                continue
            if not c.isspace():
                self.count += 1
                self.cjk += dr._is_cjk(c)
            if c in dr.CLAUSE and self.count >= self._limit():
                self._cut(i + 1, out)
                continue
            self.i = i + 1
        return out

    def close(self):
        out = []
        if self.buf.strip():
            out.append(self.buf)
            self.n += 1
        self.buf, self.i = "", 0
        return out


# ------------------------------------------------------------------ hub

def wav_bytes(x, sr):
    buf = io.BytesIO()
    pcm = (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).round()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


class Hub:
    """What the server hands out: the latest frames to each video client
    (a short queue each; a slow client loses its oldest frames, never
    holds the render back), events to each event client, audio clips by
    id. New event clients get the recent events still to come (audio and
    subtitles not yet over) and the last log lines."""

    def __init__(self, clock, max_audio=64):
        self.clock = clock
        self.lock = threading.Lock()
        self.frame_subs, self.event_subs = set(), set()
        # VRM body states (flow.vrm), one per frame, for /states.
        self.state_subs = set()
        self.last_state = None
        self.audio = OrderedDict()
        self.max_audio = max_audio
        self.history = deque(maxlen=200)
        self.last_frame = None
        self.n_audio = 0
        self.stop = threading.Event()

    def frame(self, f, t, jpeg):
        item = (f, t, jpeg)
        with self.lock:
            self.last_frame = item
            subs = list(self.frame_subs)
        for q in subs:
            _put_latest(q, item)

    def state(self, f, t, st, avatar="vrm"):
        """A VRM body state for frame f at server time t."""
        item = {"f": f, "t": round(t, 4), "avatar": avatar, **st}
        with self.lock:
            self.last_state = item
            subs = list(self.state_subs)
        for q in subs:
            _put_latest(q, item)

    def event(self, kind, **data):
        msg = dict(data, type=kind)
        with self.lock:
            self.history.append(msg)
            subs = list(self.event_subs)
        for q in subs:
            _put_latest(q, msg)
        return msg

    def add_audio(self, x, sr, t, text=""):
        with self.lock:
            self.n_audio += 1
            aid = f"a{self.n_audio}"
            self.audio[aid] = wav_bytes(x, sr)
            while len(self.audio) > self.max_audio:
                self.audio.popitem(last=False)
        self.event("audio", id=aid, t=round(t, 4),
                   dur=round(len(x) / sr, 4), url=f"/audio/{aid}.wav",
                   text=text)
        return aid

    def subscribe(self, kind, maxsize=None):
        q = queue.Queue(maxsize=maxsize or (90 if kind in ("frames", "states")
                                            else 500))
        with self.lock:
            (self.frame_subs if kind == "frames" else self.state_subs
             if kind == "states" else self.event_subs).add(q)
            if kind == "events":
                now = self.clock.now()
                for m in self.history:
                    end = m.get("t", 0) + m.get("dur", 0) \
                        if m["type"] == "audio" else m.get("end")
                    if m["type"] in ("audio", "sub") and end is not None \
                            and end < now:
                        continue
                    _put_latest(q, m)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.frame_subs.discard(q)
            self.event_subs.discard(q)
            self.state_subs.discard(q)


def _put_latest(q, item):
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


# --------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0: a streamed body ends when the connection closes, which a
    # browser's fetch() reads as a stream.
    protocol_version = "HTTP/1.0"
    server_version = "TsukuyomiLive/1"

    def log_message(self, fmt, *args):      # quiet: the app logs itself
        pass

    @property
    def app(self):
        return self.server.app

    def _send(self, code, body, ctype="application/json", extra=()):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _stream_head(self, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        hub = self.app.hub
        try:
            if path in ("/", "/index.html"):
                # A VRM avatar (alone or beside the Live2D rig) is drawn by
                # the browser: flow/web/vrm.html.
                vrm = getattr(self.app, "vrm", None)
                with open(PAGE if vrm is None else VRM_PAGE,
                          encoding="utf-8") as fh:
                    page = fh.read()
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/favicon.ico":
                # The browser asks on its own; a 404 here was the page's
                # one console error.
                self.send_response(204)
                self.end_headers()
                return None
            if path.startswith("/web/"):
                return self._static(path[5:])
            if path == "/avatar.vrm" and getattr(self.app, "vrm", None):
                with open(self.app.vrm["path"], "rb") as fh:
                    return self._send(200, fh.read(),
                                      "application/octet-stream")
            if path == "/avatar.json" and getattr(self.app, "vrm", None):
                return self._send(200, self.app.vrm["config"])
            if path == "/states":
                return self._states()
            if path == "/clock":
                return self._send(200, {"t": self.app.clock.now()})
            if path == "/state":
                return self._send(200, self.app.state())
            if path == "/frame.jpg":
                lf = hub.last_frame
                if lf is None:
                    return self._send(404, {"error": "no frame yet"})
                return self._send(200, lf[2], "image/jpeg")
            if path.startswith("/audio/") and path.endswith(".wav"):
                data = hub.audio.get(path[7:-4])
                if data is None:
                    return self._send(404, {"error": "no such clip"})
                return self._send(200, data, "audio/wav")
            if path == "/events":
                return self._events()
            if path == "/video":
                return self._frames(raw=True)
            if path == "/mjpeg":
                return self._frames(raw=False)
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            return None

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if path != "/say":
            return self._send(404, {"error": "not found"})
        try:
            text = json.loads(raw.decode("utf-8") or "{}").get("text", "")
        except ValueError:
            text = raw.decode("utf-8", "replace")
        text = str(text).strip()
        if not text:
            return self._send(400, {"error": "empty line"})
        waiting = self.app.say(text)
        return self._send(202, {"ok": True, "queued": waiting})

    def _events(self):
        hub = self.app.hub
        q = hub.subscribe("events")
        self._stream_head("text/event-stream; charset=utf-8")
        try:
            hello = {"type": "hello", "t": self.app.clock.now(),
                     "fps": self.app.fps}
            self.wfile.write(f"data: {json.dumps(hello)}\n\n".encode())
            self.wfile.flush()
            while not hub.stop.is_set():
                try:
                    m = q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")     # finds a gone client
                    self.wfile.flush()
                    continue
                line = json.dumps(m, ensure_ascii=False)
                self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)

    def _static(self, rel):
        """flow/web files (the VRM page's modules and vendored libraries);
        nothing outside that directory."""
        full = os.path.realpath(os.path.join(WEB, rel))
        if not full.startswith(os.path.realpath(WEB) + os.sep) or \
                not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = {".js": "text/javascript", ".html": "text/html; "
                 "charset=utf-8", ".json": "application/json"}.get(
                     os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as fh:
            return self._send(200, fh.read(), ctype)

    def _states(self):
        """Server-sent VRM body states, one per frame (Hub.state)."""
        hub = self.app.hub
        q = hub.subscribe("states")
        self._stream_head("text/event-stream; charset=utf-8")
        try:
            while not hub.stop.is_set():
                try:
                    m = q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                line = json.dumps(m, separators=(",", ":"))
                self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)

    def _frames(self, raw):
        hub = self.app.hub
        q = hub.subscribe("frames")
        self._stream_head("application/octet-stream" if raw else
                          "multipart/x-mixed-replace; boundary=frame")
        try:
            while not hub.stop.is_set():
                try:
                    f, t, jpeg = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if raw:
                    self.wfile.write(FRAME_HEAD.pack(FRAME_MAGIC, f, t,
                                                     len(jpeg)) + jpeg)
                else:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, app, host="127.0.0.1", port=8765):
        self.app = app
        super().__init__((host, port), Handler)

    def start(self):
        th = threading.Thread(target=self.serve_forever, name="http",
                              daemon=True)
        th.start()
        return th

    def stop(self):
        self.app.hub.stop.set()
        self.shutdown()
        self.server_close()


# ---------------------------------------------------------------- turns

# Quiet chat: the most quiet turns in a row before a real message is waited
# for, and how often the loop looks at the clock (seconds).
QUIET_MAX = 3
QUIET_POLL = 0.5
# The live loop must never freeze silently on a backend that stalls (Ollama
# once never yielded a token and the read timeout was 300 s): seconds to the
# first token of a reply, and between tokens after it; 0 turns one off.
FIRST_TOKEN_TIMEOUT = 20.0
STALL_TIMEOUT = 30.0
# Context budget (Performer.trim_context): Ollama's window when neither
# --ollama-ctx nor /api/show says, the room kept for the reply, a safety
# margin, the characters-per-token guess before a prompt_eval_count has
# calibrated one (marker JSON and CJK are token-dense), and the share of the
# window past which a reported prompt gets a warning.
DEFAULT_CTX = 8192
REPLY_RESERVE = 1024
CTX_MARGIN = 256
GUESS_CPT = 2.0
CTX_WARN = 0.85
# Chat lines queued while she talks, read as one turn: the last BATCH_MAX.
BATCH_MAX = 5


def batch_text(lines) -> str:
    """The user turn for chat lines that arrived together: the lines, one
    per line, oldest first. The chat has no user names here, and the
    student's data has single-message turns only (flow.dataset: one to three
    short sentences), so the lines are not labelled; a single line is
    returned unchanged."""
    return "\n".join(str(x).strip() for x in lines if str(x).strip())


def _checks():
    """flow.checks, imported when first needed (its chain is heavy)."""
    try:
        from . import checks as ck
    except ImportError:
        from flow import checks as ck
    return ck


def _rules():
    """flow.rules, imported when first needed (it reads flow.checks)."""
    try:
        from . import rules as rl
    except ImportError:
        from flow import rules as rl
    return rl


class Attempt:
    """One streaming try of a turn's reply: its own queue, text and state,
    so a producer thread abandoned after a stall can never push into the
    next try's (or the next turn's) queue. `dead` is set under `lock` by the
    consumer; the producer checks it under the same lock before it feeds
    anything."""

    def __init__(self, cut):
        self.q = queue.Queue()
        self.raw = []
        self.info = {"first": None, "stop": None, "error": None}
        self.lock = threading.Lock()
        self.dead = False
        self.cut = cut
        self.last = time.monotonic()      # the last token (or the start)
        self.thread = None
        self.chars = 0                    # characters sent (note_usage)
        self.messages = None              # what this try sent (a repair pair)


TURN_DUMP = 90.0      # seconds; 0: off
HISTORY_MARKERS = 2   # --history-markers
TRIM_CHUNK = 4        # --trim-chunk


def _arm_dump(seconds):
    """faulthandler's stack dump of every thread after `seconds`, once."""
    if not seconds:
        return False
    try:
        import faulthandler
        import sys
        faulthandler.dump_traceback_later(seconds, repeat=False,
                                          file=sys.stderr)
        return True
    except (ValueError, OSError, AttributeError, RuntimeError):
        return False       # no usable stderr (pytest capture, a service)


def _disarm_dump():
    import faulthandler
    faulthandler.cancel_dump_traceback_later()


class Performer:
    """Turns, one at a time, on a worker thread: the language model's
    reply streamed and cut into sentences, each spoken and put on the
    stream as soon as it is complete (module docstring, "turn")."""

    def __init__(self, a, run, rig, stream, clock, hub, backend, tts,
                 log=print, recorder=None, memory=None):
        self.a, self.run, self.rig, self.stream = a, run, rig, stream
        self.clock, self.hub, self.backend, self.tts = clock, hub, backend, tts
        self.log, self.recorder = log, recorder
        self.fps = rig.fps
        self.messages = []
        # Memory (flow.memory, --memory): the notes block of the turn in
        # flight (one snapshot per turn: the budget counts it and every
        # try sends it), the completed exchanges as plain text for the
        # consolidator, and how many user messages the trims just dropped.
        self.memory = memory
        self.note_block = ""
        # Reminders (--reminders, flow.rules): the rule lines this turn's
        # detectors picked go into note_block, behind the memory notes.
        self.reminders = getattr(a, "reminders", "off") or "off"
        self.rules_ids = []
        self._exch = []
        self._dropped_users = 0
        if memory is not None:
            memory.log = self.emit
            memory.record = self._memory_record
            if memory.path is None and getattr(recorder, "dir", None):
                memory.path = os.path.join(recorder.dir, "memory.json")
        self.inbox = queue.Queue()
        self.rng = np.random.default_rng([int(a.seed), 1])
        self.next_frame = 0               # the first frame after the speech
        self.turns = 0
        self.busy = False
        self.thread = None
        self.lang, self.lang_set = "en", False
        # The model sees the last `history_turns` exchanges (0: all of them):
        # the student was taught sessions of a few turns, and past them it
        # copies its own earlier replies.
        self.history_turns = int(getattr(a, "history_turns", 5))
        # Chunked trimming (--trim-chunk): a trim drops several exchanges at
        # once, so the sent prefix stays the same for the turns between
        # trims and Ollama's prefix cache holds (trim_history). It is capped
        # at a third of the window, so a short window trims one at a time.
        self.trim_chunk = max(1, int(getattr(a, "trim_chunk", TRIM_CHUNK)
                                     or 1))
        # Old replies lose their markers (--history-markers N): all but the
        # last N assistant messages are sent as spoken text only, once per
        # `trim_chunk` turns (retire); 0 keeps every marker.
        self.history_markers = max(0, int(getattr(a, "history_markers",
                                                  HISTORY_MARKERS) or 0))
        self._ret_upto = 0         # messages before this index: no markers
        self._spoken_cache = {}
        # The glance reflex (--glance, glance_for): the targets for this
        # turn's first sentence, taken by it.
        self.glance_mode = getattr(a, "glance", "auto") or "auto"
        self._glance = None
        # The repair layer (--repair, flow.repair): this turn's state, and
        # what the body lets it do (computed on first use).
        self.repair_mode = getattr(a, "repair", rp.DEFAULT_MODE) \
            or rp.DEFAULT_MODE
        self._rep = None
        self._rep_body = None
        # Watchdog (turn): seconds to the first token and between tokens.
        self.first_timeout = float(getattr(a, "first_token_timeout",
                                           FIRST_TOKEN_TIMEOUT) or 0)
        self.stall_timeout = float(getattr(a, "stall_timeout",
                                           STALL_TIMEOUT) or 0)
        self.batch_max = int(getattr(a, "batch_max", BATCH_MAX) or 0)
        self.batched = 1                  # chat lines in the current turn
        self._ctx = None                  # the window, resolved on first use
        self.cpt = None                   # characters per token, calibrated
        # Head carry and the expression baseline (flow.pose): the pose
        # tokens name the angles that stay and the face they keep, and the
        # end of each reply's speech releases them.
        self.carry = bool(getattr(a, "head_carry", False))
        self.baseline = fs.baseline_from_args(a) > 0
        # Talking while the chat is quiet: quiet_after seconds after the
        # last reply's speech ends (0: never), at most quiet_max in a row
        # before a real message is waited for.
        self.quiet_after = float(getattr(a, "quiet_after", 0.0) or 0.0)
        self.quiet_max = int(getattr(a, "quiet_max", QUIET_MAX))
        self.quiet_n = 0                  # quiet turns since a real message
        self.quiet_t0 = None              # when the last turn's speech ended
        self.last_user = ""               # the last real user message
        # Body mode (--mode body): the run from director.prepare_body, and
        # a callable giving the latest rendered frame (RGBA array) for the
        # picture a vision backend gets.
        self.body = run if run.get("mode") == "body" else None
        self.snap = None
        # The framing now (flow.framing), shown to a beats-prompt model with
        # each line and moved by its [frame] markers; None: not shown (v1
        # prompt). A body whose framing cannot switch (a VRM) stays full.
        # `home` (--frame) is the framing every turn starts in: a switch
        # lasts until the end of its reply, then the camera eases back.
        self.framing = self.home = None
        if self.body is not None and pz.beats_like(
                getattr(a, "body_prompt", "v1")):
            self.framing = (getattr(a, "frame", "full")
                            if fr.can_frame(self.body["body"]) else "full")
            self.home = self.framing
        # --avatar both: the VRM follows every sentence (VRMFollower).
        self.followers = []
        # Speech-driven micro-expressions (flow.micro) on this body: its
        # state keeps the spacing across sentences. Body mode with a TTS.
        self.micro = (mc.planner_for(a, self.body["body"], self.emit)
                      if self.body is not None else None)

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="turns",
                                       daemon=True)
        self.thread.start()

    def say(self, text):
        self.quiet_n = 0                  # a real message: the count starts over
        self.inbox.put(text)
        return self.inbox.qsize() + (1 if self.busy else 0)

    def quiet_due(self):
        """True when the chat has been quiet long enough for the character
        to carry on by itself: --quiet-after seconds since the last turn's
        speech ended, nothing waiting, fewer than quiet_max quiet turns
        since a real message."""
        return bool(self.quiet_after > 0 and self.quiet_t0 is not None
                    and self.quiet_n < self.quiet_max
                    and self.inbox.empty() and not self.busy
                    and self.clock.now() >= self.quiet_t0 + self.quiet_after)

    def next_turn(self, poll=QUIET_POLL):
        """(text, quiet): the next line to answer. A real message when one
        comes; the quiet-chat line when quiet_due; None, False to stop.

        Chat that came in while she was talking is read as one turn, the
        way a streamer reads it (batch_text): every line queued by now, the
        last `batch_max` of them. A real message always wins over a quiet
        turn: quiet_due needs an empty inbox."""
        while True:
            try:
                text = self.inbox.get(timeout=poll if self.quiet_after > 0
                                      else None)
            except queue.Empty:
                if self.quiet_due():
                    self.batched = 1
                    return pz.quiet_user(self.last_user)[0], True
                continue
            if text is None:
                return None, False
            lines = [text]
            while True:
                try:
                    more = self.inbox.get_nowait()
                except queue.Empty:
                    break
                if more is None:          # stop: after this turn's lines
                    self.inbox.put(None)
                    break
                lines.append(more)
            self.batched = 1
            if len(lines) > 1:
                dropped = max(0, len(lines) - self.batch_max) \
                    if self.batch_max > 0 else 0
                self.emit(f"live: {len(lines)} chat lines came in while she "
                          "was busy; answering them as one turn"
                          + (f" (the last {self.batch_max}; {dropped} older "
                             "dropped)" if dropped else ""))
                lines = lines[dropped:]
                self.batched = len(lines)
            return batch_text(lines), False

    def _loop(self):
        while True:
            text, quiet = self.next_turn()
            if text is None:
                return
            self.busy = True
            # A turn still running after TURN_DUMP seconds prints every
            # thread's stack to stderr: a freeze names its own line (the
            # edge-tts one was found with py-spy).
            dump = _arm_dump(TURN_DUMP)
            try:
                self.turn(text, quiet)
            except Exception as exc:  # noqa: BLE001 - one bad turn, not the show
                self.log(f"live: turn failed: {exc!r}")
                self.hub.event("log", text=f"turn failed: {exc}")
            finally:
                if dump:
                    _disarm_dump()
                self.quiet_t0 = max(self.next_frame / self.fps,
                                    self.clock.now())
                self.busy = False

    def emit(self, text):
        self.log(text)
        self.hub.event("log", text=text)

    def _memory_record(self, rec):
        """A memory consolidation (flow.memory) into turns.jsonl."""
        if self.recorder is not None:
            self.recorder.turn(rec)

    def _take_folds(self):
        """[(viewer line, spoken text)] of the oldest exchanges not yet
        handed over, as many as the trims dropped user messages."""
        k, self._dropped_users = self._dropped_users, 0
        out = []
        for e in self._exch:
            if k <= 0:
                break
            if not e["gone"]:
                e["gone"] = True
                out.append((e["u"], e["s"]))
                k -= 1
        while self._exch and self._exch[0]["gone"]:
            self._exch.pop(0)
        return out

    # One turn ---------------------------------------------------------

    def turn(self, text, quiet=False):
        clock = self.clock
        self.turns += 1
        n = self.turns
        t_in = clock.now()
        self.hub.event("user", text=text, t=t_in, **({"quiet": True}
                                                     if quiet else {}),
                       **({"batched": self.batched} if self.batched > 1
                          and not quiet else {}))
        if quiet:
            # Nobody wrote: the chat's language is still the last message's.
            self.quiet_n += 1
            self.lang_set = False
            self.emit(f"live: turn {n}: chat quiet "
                      f"{t_in - (self.quiet_t0 or t_in):.1f}s after the last "
                      f"reply; quiet turn {self.quiet_n}/{self.quiet_max}")
        else:
            self.last_user = text
            # The reply's language, for the voice: the prompt asks for the
            # user's language; the first sentence whose script settles it
            # wins (lang_of), and the rest of the turn keeps it.
            self.lang, self.lang_set = sp.language(text), False
        if self.body is None:
            self.messages.append({"role": "user", "content": text})
        else:
            self.messages.append(self.body_message(
                text, self.last_user if quiet else None))
        prev_reply = ""
        for m in reversed(self.messages[:-1]):
            if m.get("role") == "assistant":
                prev_reply = m.get("content") or ""
                break
        self.note_block = self.tail_block(text, prev_reply, quiet, n)
        self._dropped_users = 0
        self.trim_history(n)
        self.trim_context(n)
        self._retire()
        if self.memory is not None:
            self.memory.note_dropped(self._take_folds())
        self._glance = None if quiet else self.glance_for(text)
        self._rep = self.repair_for(text, quiet, n)
        lat, k = [], 0
        at = None
        for tries in (1, 2):
            at = self.attempt(t_in)
            status = "ok"
            while True:
                limit = (self.first_timeout if at.info["first"] is None
                         else self.stall_timeout)
                try:
                    if limit > 0:
                        item = at.q.get(timeout=max(
                            0.0, limit - (time.monotonic() - at.last)))
                    else:
                        item = at.q.get()
                except queue.Empty:
                    if limit > 0 and time.monotonic() - at.last >= limit:
                        status = ("first" if at.info["first"] is None
                                  else "stall")
                        break
                    continue
                if item is None:
                    break
                s, t_cut = item
                k += 1
                if lat and not mem.strip_markers(s):
                    self.trailing(s, n, k, t_in, t_cut)
                    k -= 1
                    continue
                res = self.sentence(s, n, k, t_in, t_cut)
                if res:
                    lat.append(res)
            if status == "ok":
                at.thread.join()
                self.note_usage(n, at.chars)
                break
            rest = self.abandon(at)
            if status == "stall":
                # Mid-stream: what was said stays (history, speech); the
                # words still held by the cutter are spoken as well.
                self.emit(f"live: turn {n}: no token from the backend for "
                          f"{self.stall_timeout:g} s mid-reply; ending the "
                          "reply here")
                at.info["error"] = TimeoutError(
                    f"stalled {self.stall_timeout:g} s mid-stream")
                for s in rest:
                    k += 1
                    if lat and not mem.strip_markers(s):
                        self.trailing(s, n, k, t_in, self.clock.now() - t_in)
                        k -= 1
                        continue
                    res = self.sentence(s, n, k, t_in, self.clock.now() - t_in)
                    if res:
                        lat.append(res)
                break
            if tries == 1:
                self.emit(f"live: turn {n}: no token from the backend in "
                          f"{self.first_timeout:g} s; retrying")
                continue
            self.emit(f"live: turn {n}: no token from the backend in "
                      f"{self.first_timeout:g} s again; dropping this turn")
            at.info["error"] = TimeoutError(
                f"no first token in {self.first_timeout:g} s, twice")
        raw, info = at.raw, at.info
        reply = raw_reply = "".join(raw)
        if self._rep is not None:
            # The reply as she performed it is what the history keeps.
            for f in self._rep.finish():
                self.report_repair(n, f)
            reply = self._rep.rebuild(reply)
        if self.framing is not None and self.framing != self.home:
            # The switch lasts for this reply: back to home at the end of
            # its speech, so the next turn starts, and is shown, there.
            t_back = max(self.next_frame / self.fps, self.clock.now())
            self.stream.direct([fr.frame_token(self.home, t_back)])
            self.emit(f"live: turn {n} camera -> {self.home} at "
                      f"{t_back:.2f}s (home)")
            self.framing = self.home
        if self.body is not None and (self.carry or self.baseline):
            # The reply's speech is over: the head and gaze it held relax
            # (over at least a second) and its resting face fades.
            t_end = max(self.next_frame / self.fps, self.clock.now())
            self.stream.direct([pz.release_token(t_end)])
            for fol in self.followers:
                fol.release(t_end)
        if self.body is not None:
            self.messages[-1] = pz.text_only(self.messages[-1])
        if info["error"] is not None:
            self.emit(f"live: turn {n}: model failed: {info['error']!r}")
        refused = info["stop"] == "refusal"
        if refused or not reply.strip():
            self.messages.pop()                 # not history, as director
            self.emit(f"live: turn {n}: "
                      + ("refused" if refused else "no reply"))
        else:
            self.messages.append({"role": "assistant", "content": reply})
            if self.memory is not None:
                self._exch.append({"u": "" if quiet else text,
                                   "s": mem.strip_markers(reply),
                                   "gone": False})
                live_ex = [e for e in self._exch if not e["gone"]]
                self.memory.after_turn(
                    n, text, reply, prev_reply, quiet,
                    [(e["u"], e["s"]) for e in live_ex[-mem.RECENT:]])
        first = info["first"]
        summary = {"turn": n, "user": text, "reply": reply,
                   **({"quiet": True} if quiet else {}),
                   **({"batched": self.batched} if self.batched > 1
                      and not quiet else {}),
                   "model_first_text": None if first is None
                   else round(first, 3), "sentences": lat,
                   "stop_reason": info["stop"]}
        if lat:
            summary["first_audio_after"] = lat[0]["audio_at"]
        if self.framing is not None:
            summary["framing"] = self.framing
        if reply != raw_reply:
            summary["reply_unrepaired"] = raw_reply
            summary["repaired"] = [f["check"] for f in self._rep.fixes
                                   if f["applied"]]
            if not (refused or not reply.strip()):
                self.repair_pair(n, text, raw_reply, reply, at)
        self.hub.event("reply", **summary)
        if self.recorder is not None:
            self.recorder.turn(summary)
        self.emit(f"live: turn {n}: model first text "
                  + (f"{first:.2f}s" if first is not None else "never")
                  + (f", first audio {lat[0]['audio_at']:.2f}s after enter"
                     if lat else "") + f"; reply: {reply.strip()}")

    def tail_block(self, text, prev_reply, quiet, n) -> str:
        """The block that rides the latest user message this turn: the
        memory notes, then flow.rules' "(reminders: ...)" line when
        --reminders is auto (body mode: the rules are about markers). One
        snapshot per turn, as the budget counts it and every try sends it."""
        notes = self.memory.block() if self.memory else ""
        self.rules_ids = []
        if self.reminders != "auto" or self.body is None:
            return notes
        rl = _rules()
        cur = self.memory.current() if self.memory is not None else None
        if getattr(self, "_rule_can", None) is None:
            self._rule_can = rl.capabilities(self.body["body"])
        out, ids = rl.tail_block(text, prev_reply, notes, cur,
                                 self._rule_can, "auto")
        if ids:
            self.rules_ids = ids
            self.emit(f"rules: turn {n}: {', '.join(ids)}")
        return out

    def sent_chars(self, messages=None) -> int:
        """Characters of the request: the system prompt and the messages."""
        msgs = self.sent_view() if messages is None else messages
        # The turn's notes block goes into the request at send time: the
        # history estimate counts it; a request already built has it.
        extra = len(self.note_block) + 1 if (messages is None
                                              and self.note_block) else 0
        return len(self.run["system"]) + extra + sum(
            len(m["content"]) if isinstance(m.get("content"), str)
            else len(str(m.get("content"))) for m in msgs)

    def context_window(self) -> int:
        """num_ctx the model runs with: --ollama-ctx, else the backend's own
        (Ollama /api/show), else DEFAULT_CTX. Resolved once."""
        if self._ctx is None:
            ctx = int(getattr(self.a, "ollama_ctx", None) or 0)
            if not ctx:
                fn = getattr(self.backend, "ctx_window", None)
                try:
                    ctx = int(fn() or 0) if fn is not None else 0
                except Exception:  # noqa: BLE001
                    ctx = 0
            self._ctx = ctx or DEFAULT_CTX
        return self._ctx

    def prompt_budget(self) -> int:
        """Tokens the request may take: the window minus the reply's room
        (REPLY_RESERVE, or --max-tokens when smaller) and a margin."""
        mt = getattr(self.a, "max_tokens", None)
        reserve = min(REPLY_RESERVE, int(mt)) if mt else REPLY_RESERVE
        return self.context_window() - reserve - CTX_MARGIN

    def trim_context(self, n):
        """Drop the oldest exchanges until the estimated prompt fits
        prompt_budget (after the --history-turns cap): never the system
        prompt or the current user message. Tokens are estimated from the
        chars-per-token ratio the last reported prompt_eval_count gave
        (note_usage), GUESS_CPT before one."""
        cpt = self.cpt or GUESS_CPT
        budget, gone = self.prompt_budget(), 0
        est = self.sent_chars() / cpt
        while est > budget and len(self.messages) > 1:
            k = 2 if len(self.messages) > 2 else 1
            self._drop_front(k)
            gone += k
            est = self.sent_chars() / cpt
        if gone and self.trim_chunk > 1:
            # A chunk, not one exchange: room for the next few turns, so
            # the sent prefix holds (trim_history).
            per = est / max(1, len(self.messages) // 2)
            target = budget - (self.trim_chunk - 1) * per
            while est > target and len(self.messages) > 3:
                self._drop_front(2)
                gone += 2
                est = self.sent_chars() / cpt
        if gone:
            self._retire(force=True)
            self.emit(f"live: turn {n}: history trimmed for the context "
                      f"window ({gone} older messages not sent, a chunk of "
                      f"up to {self.trim_chunk} exchanges; context "
                      f"budget: ~{int(est)} of {budget} tokens, window "
                      f"{self.context_window()})")

    def note_usage(self, n, chars):
        """After a reply: calibrate chars per token from the backend's
        prompt_eval_count for the `chars` it was sent, and warn when the
        prompt used most of the window. The ratio only ever goes down
        (Ollama may leave cached prompt tokens out of the count, which
        would make text look cheaper than it is)."""
        pc = (getattr(self.backend, "usage", None) or {}).get(
            "prompt_eval_count")
        if not isinstance(pc, (int, float)) or pc <= 0 or chars <= 0:
            return
        r = max(1.0, min(chars / pc, 4.0))
        self.cpt = r if self.cpt is None else min(self.cpt, r)
        ctx = self.context_window()
        if pc > CTX_WARN * ctx:
            self.emit(f"live: turn {n}: WARNING prompt was {int(pc)} tokens "
                      f"of a {ctx}-token window ({pc / ctx:.0%}); older "
                      "history will be trimmed, or raise --ollama-ctx")

    def attempt(self, t_in):
        """Start one streaming try of the reply on its own thread. The
        thread works on a copy of the messages and feeds only its own
        Attempt, so a try abandoned by the watchdog cannot touch the next
        one or the history."""
        at = Attempt(SentenceCutter(max_tag=self.run.get("max_tag")))
        messages = self.sent_view()
        if self.note_block:
            messages = mem.inject(messages, self.note_block,
                                  getattr(self.memory, "place", "tail"))
        at.chars = self.sent_chars(messages)
        at.messages = messages
        clock = self.clock

        def produce():
            gen = None
            try:
                gen = iter(self.backend.stream(self.run["system"], messages))
                for chunk in gen:
                    with at.lock:
                        if at.dead:
                            return
                        at.last = time.monotonic()
                        if at.info["first"] is None:
                            at.info["first"] = clock.now() - t_in
                        at.raw.append(chunk)
                        for s in at.cut.feed(chunk):
                            at.q.put((s, clock.now() - t_in))
                with at.lock:
                    if not at.dead:
                        for s in at.cut.close():
                            at.q.put((s, clock.now() - t_in))
            except Exception as exc:  # noqa: BLE001
                with at.lock:
                    if not at.dead:
                        at.info["error"] = exc
            finally:
                with at.lock:
                    if not at.dead:
                        at.info["stop"] = getattr(self.backend, "stop_reason",
                                                  None)
                        at.q.put(None)
                close = getattr(gen, "close", None)
                if close is not None:
                    try:
                        close()           # leaves the HTTP response's `with`
                    except Exception:  # noqa: BLE001
                        pass

        at.thread = threading.Thread(target=produce, name="llm", daemon=True)
        at.thread.start()
        return at

    def abandon(self, at):
        """Give up on a try that stalled: mark it dead (its thread feeds
        nothing from now on), close the backend's connection where it can
        (a blocked read then raises and the thread ends), and return the
        sentences its cutter still held plus any unread in its queue."""
        with at.lock:
            at.dead = True
            at.info["stop"] = "stalled"
            held = at.cut.close()
        out = []
        while True:
            try:
                item = at.q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                out.append(item[0])
        cancel = getattr(self.backend, "cancel", None)
        if cancel is not None:
            try:
                cancel()
            except Exception as exc:  # noqa: BLE001
                self.log(f"live: backend cancel failed: {exc!r}")
        return out + list(held)

    def _drop_front(self, k):
        """Delete the oldest k messages for good (counting the user
        messages for the memory fold and moving the marker boundary)."""
        self._dropped_users += sum(
            m.get("role") == "user" for m in self.messages[:k])
        del self.messages[:k]
        self._ret_upto = max(0, self._ret_upto - k)

    def spoken_of(self, raw) -> str:
        """A reply's spoken text, markers parsed out by the same parser the
        turn uses (the reply itself when nothing is spoken)."""
        c = self._spoken_cache.get(raw)
        if c is None:
            p = (dr.TagParser(self.run["labels"], self.run["offered"])
                 if self.body is None else self.body["parser"]())
            p.feed(raw)
            p.close()
            c = (p.spoken or "").strip() or raw
            if len(self._spoken_cache) > 256:
                self._spoken_cache.clear()
            self._spoken_cache[raw] = c
        return c

    def _retire(self, force=False):
        """Move the marker boundary (--history-markers N): every assistant
        message but the last N is sent as spoken text only from now on. The
        boundary moves when the messages with markers reach N + chunk (or
        at a trim, when the prefix moves anyway), never in between, so a
        message is sent with the same bytes for turns on end."""
        n = self.history_markers
        if n <= 0:
            return
        idx = [i for i, m in enumerate(self.messages)
               if m.get("role") == "assistant"]
        bearing = [i for i in idx if i >= self._ret_upto]
        if len(idx) > n and (force or len(bearing) >= n + self.trim_chunk):
            self._ret_upto = idx[-n]

    def sent_view(self):
        """The history as sent, before the notes: the stored messages with
        the assistant ones behind the marker boundary reduced to their
        spoken text. The stored history is untouched."""
        out = list(self.messages)
        for i in range(min(self._ret_upto, len(out))):
            m = out[i]
            if m.get("role") == "assistant" and isinstance(
                    m.get("content"), str):
                out[i] = {**m, "content": self.spoken_of(m["content"])}
        return out

    def trim_history(self, n):
        """Keep the system prompt and the last `history_turns` exchanges
        before this turn's user message (a request of at most
        2 * history_turns + 1 messages)."""
        keep = 2 * self.history_turns + 1
        if self.history_turns > 0 and len(self.messages) > keep:
            # Over the cap: down to cap - chunk + 1 exchanges, so the next
            # chunk - 1 turns add to the same prefix instead of moving it.
            chunk = min(self.trim_chunk, max(1, self.history_turns // 3))
            kept = self.history_turns - chunk + 1
            gone = len(self.messages) - (2 * kept + 1)
            self._drop_front(gone)
            self._retire(force=True)
            self.emit(f"live: turn {n}: history trimmed to the last "
                      f"{kept} exchanges ({gone} older messages not sent"
                      + (f"; a chunk of {chunk} exchanges, the next trim "
                         f"in about {chunk} turns, the sent prefix "
                         "unchanged until then" if chunk > 1 else "")
                      + ")")

    def trailing(self, raw, n, k, t_in, t_cut):
        """Markers after the last words of a reply (a pose with nothing
        spoken after it). With the head carry or the expression baseline
        (the end of the speech releases both at once) such a pose would
        flicker for a frame or two before it relaxed: it is dropped. Without
        them it holds, so it is merged into the previous sentence: applied
        at the end of its speech (the plan of a marker-only sentence), and
        not planned, logged or counted as a sentence of its own."""
        if self.body is not None and (self.carry or self.baseline):
            self.emit(f"live: turn {n}.{k} trailing markers with no speech "
                      "dropped (the release at the end of the reply would "
                      f"make them a flicker): {raw.strip()}")
            if self._rep is not None:
                self._rep.skip(raw)
            return None
        return self.sentence(raw, n, k, t_in, t_cut, trailing=True)

    # The repair layer (flow.repair) ------------------------------------

    def repair_for(self, text, quiet, n):
        """This turn's flow.repair.Turn, or None (off, or not body mode).
        The body gives what a fix may rely on: whether the camera switches,
        whether the hands reach anything and whether [move wave] parses."""
        if self.body is None or self.repair_mode == "off":
            return None
        if self._rep_body is None:
            bm = self.body["body"]
            wave = False
            try:
                p = self.body["parser"]()
                p.feed(rp.WAVE_MARKER)
                p.close()
                wave = any(t["tag"] == "move"
                           and (t["arg"] or {}).get("name") == "wave"
                           for t in p.tags)
            except Exception:  # noqa: BLE001 - no wave to add then
                pass
            try:
                frames = bool(fr.can_frame(bm))
            except Exception:  # noqa: BLE001
                frames = False
            name = os.path.basename(str(self.body.get("body_path") or "?"))
            self._rep_body = {"wave": wave, "frames": frames,
                              "hands": rp.hands_of(bm),
                              "avatar": name.split(".")[0] or "?"}
        b = self._rep_body
        return rp.Turn(text, self.repair_mode, b["avatar"],
                       start=self.framing, frames=b["frames"],
                       hands=b["hands"], wave=b["wave"], quiet=quiet, turn=n)

    def report_repair(self, n, f):
        """A fix into the log, the page's log and turns.jsonl."""
        self.emit(rp.log_line(n, f))
        if self.recorder is not None:
            self.recorder.turn(rp.record(n, self._rep.user, f,
                                         self.repair_mode))

    def repair_pair(self, n, text, raw_reply, reply, at):
        """The DPO pair of a repaired turn (flow.repair.pair) into
        <record>/repair_pairs.jsonl: the prompt as sent, chosen = what she
        performed, rejected = what the model wrote, when the raw reply fails
        the check the fix is for and the repaired one passes it."""
        folder = getattr(self.recorder, "dir", None)
        if self.repair_mode != "auto" or not folder or not at.messages:
            return
        check, verdicts = self._rep.pair_check(raw_reply, reply)
        if check is None:
            return
        prompt = [{"role": "system", "content": self.run["system"]}] + [
            pz.text_only(m) for m in at.messages]
        # The trainer's session is the id without its last part (it holds
        # whole sessions out for its eval): one session per turn, or a
        # recording's pairs would all be held out together.
        rid = "live/" + os.path.basename(os.path.normpath(folder))
        row, meta = rp.pair(prompt, raw_reply, reply, check, {
            "id": f"{rid}/t{n:02d}/repair", "avatar": self._rep.avatar,
            "turn": n, "lang": pz.detect_lang(text),
            "checks": sorted(verdicts),
            "verdicts": {c: list(v) for c, v in verdicts.items()},
            "fixes": [{"check": f["check"], "before": f["before"],
                       "after": f["after"]} for f in self._rep.fixes
                      if f["applied"]]})
        rp.write_pair(folder, row, meta)
        self.emit(f"repair: turn {n} pair -> repair_pairs.jsonl "
                  f"({meta['kind']})")

    def glance_for(self, text):
        """The glance reflex for a user line (--glance auto): {"place",
        "targets", "why"} when the line puts something near her without
        asking her to look (flow.checks.mentioned_place), with the targets
        of the glance among the variables her body has; else None."""
        if self.glance_mode != "auto" or self.body is None:
            return None
        ck = _checks()
        place = ck.mentioned_place(text)
        if place is None:
            return None
        have = set(pz.available(self.body["body"]))
        targets = {k: v for k, v in ck.glance_targets(place).items()
                   if k in have}
        if not targets:
            return None
        why = ", ".join(x for x in (
            "behind" if place["behind"] else "",
            {-1: "her right", 1: "her left"}.get(place["side"], ""),
            {-1: "down", 1: "up"}.get(place["vert"], "")) if x)
        return {"place": place, "targets": targets, "why": why}

    def sentence(self, raw, n, k, t_in, t_cut, trailing=False):
        """Plan one sentence and put it on the stream. Returns its
        latencies (seconds after the line was entered)."""
        a, fps = self.a, self.fps
        if self._rep is not None:
            # Before the markers are planned: this sentence only, never one
            # already spoken (flow.repair).
            raw, fixes = self._rep.sentence(raw, k)
            for f in fixes:
                self.report_repair(n, f)
        p = (dr.TagParser(self.run["labels"], self.run["offered"])
             if self.body is None else self.body["parser"]())
        p.feed(raw)
        p.close()
        for w in p.warnings:
            self.emit(f"director: {w}")
        spoken, tags = p.spoken, p.tags
        t0 = self.clock.now()
        mood = "neutral"
        audio = None
        sp_ends = None
        said = None
        if spoken and self.tts is not None:
            if getattr(a, "mood", "auto") != "off":
                mood = sp.mood_of(spoken, tags)[0]
            said = None
            for tries in (1, 2):
                try:
                    said = sp.speak(spoken, tags, self.tts, self._weight,
                                    log=self.log, moods=[(0, mood)],
                                    lang=self.lang_of(spoken))
                    break
                except Exception as exc:  # noqa: BLE001 - TTS must not stop
                    self.log(f"live: turn {n}.{k} tts failed "
                             f"({type(exc).__name__}: {exc}); "
                             + ("retrying" if tries == 1 else
                                "this sentence goes without a voice"))
        if said is not None:
            acts, spans, end = said["acts"], said["spans"], said["end"]
            audio, sr, moras = said["audio"], said["sr"], said.get("moras")
            sp_ends = mc.sentence_ends(spoken, spans) or None
        elif spoken:
            acts, spans, end = dr.timeline(spoken, tags, a.cjk_rate,
                                           a.latin_rate)
        else:
            acts = [{"t": 0.0, "tag": t["tag"], "arg": t["arg"], "at": 0}
                    for t in tags if t["tag"] != "pause"]
            spans, end = [], 0.0
        t_tts = self.clock.now() - t0
        mouth = None
        if audio is not None and (a.lipsync or "auto") != "off":
            nf = int(math.ceil(end * fps)) + 2
            lip = a.lipsync or "auto"
            cal = getattr(a, "mouth_cal", None) if lip == "auto" else None
            mouth = sp.mouth_track(audio, sr, fps, nf, lip, a.mouth_gain,
                                   cal=cal, moras=moras)
        mvs = []
        if self.body is not None and any(x["tag"] == "move" for x in acts):
            # Motion primitives (flow.moves), built from where the body is
            # heading, before the sentence's time is taken (a 16-beat dance
            # is 0.6-1 s of solves; after it, the audio would start late).
            # Each marker's variation is seeded by the run's seed, the turn
            # and the sentence, written on the marker: the --avatar both
            # follower draws the same, and the realize generator is not
            # touched.
            t_m = self.clock.now()
            base = mv.move_seed(a, 1000 * n + k)
            for i, x in enumerate(acts):
                if x["tag"] == "move":
                    x["arg"] = dict(x["arg"], seed=mv.hash_seed(
                        base, x.get("at"), i))
            cur, cur_parts = self.stream.body_now()
            mvs = mv.solve_moves(self.body["body"], acts, cur, cur_parts,
                                 log=self.log, seed=base)
            mv.drop_clashing_gestures(acts, mvs, self.rig.motion_seconds,
                                      self.log)
            for fol in self.followers:
                # The --avatar both VRM builds its own now too, so its
                # dance starts with this one.
                fol.build_moves(acts, base)
            lat_move = self.clock.now() - t_m
        # When: as soon as possible, never over the previous sentence.
        now = self.clock.now()
        lead_f = int(math.ceil((now + a.lead) * fps))
        if spoken:
            f0 = max(lead_f, self.next_frame)
        else:
            # Tags after the last words: at the end of that speech.
            f0 = max(lead_f, self.next_frame - int(round(a.gap * fps)))
        t_start = f0 / fps
        late = (f0 - self.next_frame) / fps if spoken and k > 1 else 0.0
        if spoken:
            self.next_frame = f0 + int(math.ceil((end + a.gap) * fps))
        g, self._glance = self._glance, None
        if g is not None and self.body is not None and not trailing:
            ck = _checks()
            first = next((x for x in acts if x.get("tag") == pz.POSE), None)
            tg1 = ((first.get("arg") or {}).get("targets") or {}) \
                if first is not None else {}
            if ck.looks_toward(tg1, g["place"]):
                self.emit(f"glance: turn {n} skipped ({g['why']}): her "
                          "first pose already looks that way")
            else:
                # An inserted beat before her first pose: the pose solver
                # starts her own pose on the same variables after its hold.
                new = [{"t": 0.0, "tag": pz.POSE, "at": 0,
                        "arg": {"targets": dict(g["targets"]),
                                "dur": ck.GLANCE_DUR,
                                "hold": ck.GLANCE_HOLD}}]
                back = {v: 0 for v in g["targets"] if v not in tg1}
                if self.carry and back:
                    # A carried angle stays until it is moved: send it home.
                    new.append({"t": 0.0, "tag": pz.POSE, "at": 0,
                                "arg": {"targets": back, "dur": 0.3,
                                        "hold": 0.0}})
                acts[0:0] = new
                self.emit(f"glance: turn {n} at {g['why']}: "
                          + json.dumps(g["targets"]) + f" for "
                          f"{ck.GLANCE_DUR + ck.GLANCE_HOLD:.2f}s, then "
                          + ("home" if self.carry and back else
                             "her own first pose"))
        poses = []
        if self.body is not None and any(x["tag"] == pz.POSE for x in acts):
            bm = self.body["body"]
            t_s = self.clock.now()
            cur, cur_parts = self.stream.body_now()
            poses = pz.solve_poses(bm, acts, cur, cur_parts, log=self.log,
                                   keep_shown=True)
            arms = self.body.setdefault("arm_params", pz.arm_params(bm))
            for rec in poses:
                rec["arm_params"] = sorted(set(rec["params"]) & arms)
            # Solved first: a gesture under an arm pose gives way to it
            # before anything is realized (--body-gestures).
            pz.drop_clashing_gestures(acts, poses, self.rig.motion_seconds,
                                      self.log)
        shifted = [dict(x, t=round(float(x["t"]) + t_start, 4)) for x in acts]
        directed = dr.realize(shifted, self.rig.stats, list(self.rig.roles),
                              self.rig.motion_seconds, self.rng,
                              self.stream.pose(), settle=self.rig.settle)
        if self.framing is not None:
            # [frame] markers: the camera's ease at the marker's time; the
            # next turn's camera line shows where it ends.
            fts = fr.frame_tokens(shifted)
            if getattr(self.a, "camera_follow", True) and mvs:
                # Whole-body moves in bust (flow.framing.follow_tokens).
                fts = fts + fr.follow_tokens(mvs, fts, self.framing, t_start,
                                             log=self.emit)
            if fts:
                directed += fts
                self.framing = fr.last_frame(acts, self.framing)
                self.emit(f"live: turn {n}.{k} camera -> "
                          + ", ".join(f"{x['view']} at {x['t']:.2f}s"
                                      for x in fts))
        if poses:
            for k_ in pz.tokens(poses, self.rig.roles, arms,
                                carry=self.carry, baseline=self.baseline):
                k_["t"] = round(float(k_["t"]) + t_start, 4)
                directed.append(k_)
            for rec in poses:                  # stream time, for the log
                rec["start"] = round(rec["start"] + t_start, 4)
            # A gesture an earlier sentence directed gives way too.
            arm_w = [(p_["start"], p_["start"] + p_["dur"] + p_["hold"]
                      + p_["release"]) for p_ in poses
                     if p_.get("arm_params") or p_.get("parts")]
            if arm_w and self.stream.drop_gestures(arm_w):
                self.emit(f"live: turn {n}.{k} an earlier gesture gives way "
                          "to an arm pose")
            lat_solve = self.clock.now() - t_s
        if mvs:
            for k_ in mv.tokens(mvs):
                k_["t"] = round(float(k_["t"]) + t_start, 4)
                directed.append(k_)
            for rec in mvs:
                rec["start"] = round(rec["start"] + t_start, 4)
            if self.stream.drop_gestures(mv.windows(mvs)):
                self.emit(f"live: turn {n}.{k} an earlier gesture gives way "
                          "to a motion")
            for line in mv.report_lines(mvs, spoken):
                self.emit(f"live: turn {n}.{k} {line}")
        if self.micro is not None and audio is not None:
            # Between the model's beats: from this sentence's audio, off
            # the variables its pose markers touch (flow.micro).
            directed += self.micro.plan(audio, sr, acts, t_start,
                                        1000 * n + k, sp_ends)
        gone = self.stream.direct(directed) if directed else 0
        if mouth is not None:
            self.stream.add_mouth(f0, mouth, len(mouth["open"]))
        for fol in self.followers:
            fol.sentence(acts, t_start, f0, spoken,
                         None if audio is None else (audio, sr, moras),
                         end)
        text = spoken
        if audio is not None:
            self.hub.add_audio(audio, sr, t_start, text)
            if self.recorder is not None:
                self.recorder.audio(t_start, audio, sr)
        for s in dr.subtitles(spoken, spans) if spoken else ():
            sub = {"start": round(s["start"] + t_start, 3),
                   "end": round(s["end"] + t_start, 3), "text": s["text"]}
            self.hub.event("sub", t=sub["start"], end=sub["end"],
                           text=s["text"])
            if self.recorder is not None:
                self.recorder.sub(sub)
        voice = (self.tts.mood_voice(mood)
                 if self.tts is not None and mood != "neutral"
                 and hasattr(self.tts, "mood_voice") else None)
        tags_s = " ".join(pz.tag_text(x) for x in acts)
        lat = {"k": k, "cut_at": round(t_cut, 3),
               "tts": round(t_tts, 3),
               "audio_at": round(t_start - t_in, 3),
               "speech": round(end, 3), "mood": mood,
               "gap_late": round(max(late, 0.0), 3)}
        if trailing and not spoken:
            self.emit(f"live: turn {n}.{k} trailing markers merged at the "
                      f"end of the previous sentence's speech; {tags_s}")
            return lat
        self.emit(f"live: turn {n}.{k} cut {t_cut:.2f}s, "
                  f"{'tts' if audio is not None else 'plan'} {t_tts:.2f}s, "
                  f"audio at {t_start - t_in:.2f}s ({end:.2f}s), mood {mood}"
                  + (f" ({voice})" if voice else "")
                  + (f", {late:.2f}s after the previous" if late > 0.05
                     else "")
                  + (f"; {tags_s}" if tags_s else "")
                  + (f"; {gone} idle tokens displaced" if gone else "")
                  + f": {spoken}")
        if mvs:
            lat["moves"] = round(lat_move, 3)
        if poses:
            # The exact check after the poses are on the stream: the Core's
            # geometry is a report, not something the motion waits for.
            pz.add_exact(self.body["body"], self.body.get("core"), poses,
                         self.log)
            for line in pz.report_lines(poses, spoken):
                self.emit(f"live: turn {n}.{k} {line}")
            lat["solve"] = round(lat_solve, 3)
            lat["reach"] = pz.reach_summary(poses)
        return lat

    def body_message(self, text, lang_text=None):
        """The user line with where the body is now and, for a backend that
        takes images (--body-image), the frame on screen. lang_text: the
        message the reply language is read from when this line is not one
        (a quiet turn)."""
        bm = self.body["body"]
        vals, parts = self.stream.body_now()
        img = None
        if self.body.get("images") and self.snap is not None:
            frame = self.snap()
            if frame is not None:
                from PIL import Image
                buf = io.BytesIO()
                Image.fromarray(np.asarray(frame)).convert("RGB").save(
                    buf, "PNG")
                img = buf.getvalue()
        tail = (pz.turn_tail(bm, self.framing, text,
                             getattr(self.a, "body_prompt", "v1"),
                             lang_text=lang_text)
                if self.framing is not None else "")
        return pz.user_message(text + "\n" + pz.pose_now(bm, vals, parts)
                               + tail, self.a.backend, img)

    def lang_of(self, spoken):
        """The language to speak this sentence in. Offline, the voice is
        picked from the whole reply; here a sentence comes alone, and
        "本当？" (no kana) reads as Chinese, "OK！" as English. Kana or
        Hangul settle it whenever they appear; enough Latin letters, or
        Han alone (Chinese unless the turn is Japanese), settle it only
        while nothing has; anything else keeps the turn's language."""
        s = sp.script_language(spoken)
        if s in ("ja", "ko"):
            self.lang, self.lang_set = s, True
        elif not self.lang_set and s == "en":
            self.lang, self.lang_set = "en", True
        elif not self.lang_set and s == "han":
            self.lang = "ja" if self.lang == "ja" else "zh"
            self.lang_set = True
        return self.lang

    def _weight(self, ch):
        return 1.0 / (self.a.cjk_rate if dr._is_cjk(ch) else self.a.latin_rate)


# ------------------------------------------------------- render, encode

class Encoder:
    """JPEG-encodes rendered frames on its own thread (Pillow releases the
    GIL while it encodes) and hands them to the hub and the recorder. A
    full queue drops the frame rather than hold the render loop."""

    def __init__(self, hub, quality=80, recorder=None, log=print):
        self.hub, self.quality, self.recorder, self.log = hub, quality, \
            recorder, log
        self.q = queue.Queue(maxsize=4)
        self.dropped = 0
        self.errors = 0
        self.encode_s = deque(maxlen=300)
        self.thread = threading.Thread(target=self._loop, name="jpeg",
                                       daemon=True)
        self.thread.start()

    def submit(self, f, t, img):
        try:
            self.q.put_nowait((f, t, img))
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=10)

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            f, t, img = item
            try:
                t0 = time.perf_counter()
                jpeg = encode_jpeg(img, self.quality)
                self.encode_s.append(time.perf_counter() - t0)
                self.hub.frame(f, t, jpeg)
                if self.recorder is not None:
                    self.recorder.frame(f, jpeg)
            except Exception as exc:  # noqa: BLE001 - say so, keep going
                self.errors += 1
                if self.errors <= 3:
                    self.log(f"live: frame {f} not encoded: {exc!r}")


def encode_jpeg(img, quality=80):
    from PIL import Image
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a), "RGB").save(
        buf, "JPEG", quality=int(quality))
    return buf.getvalue()


class FakeBackend:
    """A renderer with no GL: a flat image whose colour follows the head
    and mouth, so the pipeline (and the page) can run anywhere. Used by the
    tests and --renderer none."""

    def __init__(self, size=128, cost=0.0):
        self.size, self.cost = size, cost
        self.dt = None
        self.n = 0
        self.views = []                    # set_view calls, in order

    def set_view(self, scale=1.0, offset_x=0.0, offset_y=0.0):
        self.views.append((scale, offset_x, offset_y))

    def render(self, overrides=None, parts=None):
        if self.cost:
            time.sleep(self.cost)
        self.n += 1
        o = overrides or {}
        img = np.zeros((self.size, self.size, 4), np.uint8)
        img[..., 3] = 255
        img[..., 0] = int(127 + 4 * float(o.get("ParamAngleX", 0.0))) % 256
        img[..., 1] = int(255 * float(o.get("ParamMouthOpenY", 0.0))) % 256
        img[..., 2] = (self.n * 3) % 256
        return img


class RenderLoop:
    """Frame f at f / fps on the clock: compute its parameters, render,
    hand to the encoder. Late frames are stepped but not rendered; the next
    render's dt covers them."""

    def __init__(self, stream, backend, clock, encoder, fps, log=print,
                 absent=(), report=10.0):
        self.stream, self.backend, self.clock = stream, backend, clock
        self.encoder, self.fps, self.log = encoder, float(fps), log
        self.absent = set(absent)
        self.report = report
        self.rendered = self.skipped = 0
        self.last_img = None                   # body mode's self-image
        self.view = None                       # the last set_view state
        self.f = 0
        self.cost = deque(maxlen=300)          # seconds per rendered frame

    def run(self, stop, max_frames=None):
        dt0 = 1.0 / self.fps
        t_rep, n_rep, s_rep = self.clock.now(), 0, 0
        while not stop.is_set():
            if max_frames is not None and self.f >= max_frames:
                return
            now = self.clock.now()
            due = int(now * self.fps)
            if self.f > due:
                time.sleep(min(self.f / self.fps - now, 0.05))
                continue
            skip = due - self.f
            if max_frames is not None:
                skip = min(skip, max_frames - 1 - self.f)
            for _ in range(skip):
                self.stream.frame()             # stateful: never skipped
            self.f += skip
            self.skipped += skip
            t0 = time.perf_counter()
            over, parts = self.stream.frame()
            if self.absent:
                over = {p: v for p, v in over.items() if p not in self.absent}
            self.backend.dt = dt0 * (skip + 1)
            cam = getattr(self.stream, "camera", None)
            if cam is not None and hasattr(self.backend, "set_view"):
                v = cam.at(self.f / self.fps)
                if v != self.view:
                    self.backend.set_view(*v)
                    self.view = v
            img = self.backend.render(over, parts=parts)
            self.last_img = img
            self.backend.dt = dt0
            self.cost.append(time.perf_counter() - t0)
            self.encoder.submit(self.f, self.f / self.fps, img)
            self.rendered += 1
            self.f += 1
            now = self.clock.now()
            if self.report and now - t_rep >= self.report:
                fps = (self.rendered - n_rep) / (now - t_rep)
                self.log(f"live: {fps:.1f} fps rendered of {self.fps:g} "
                         f"({self.skipped - s_rep} frames skipped, "
                         f"{1000 * np.median(self.cost):.1f} ms per frame "
                         f"to render, {self.encoder.dropped} encoder "
                         "drops)")
                t_rep, n_rep, s_rep = now, self.rendered, self.skipped


# ------------------------------------------------------------- recording

class Recorder:
    """--record DIR (module docstring). Frames are the encoder's JPEGs piped
    to ffmpeg (image2pipe -> H.264). Keyed on frame numbers: a frame the
    render skipped or the encoder dropped is filled with the next frame
    that arrives, so the video keeps the clock the speech is mixed on."""

    def __init__(self, out_dir, fps, log=print, video=True):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.fps, self.log = out_dir, float(fps), log
        self.lock = threading.Lock()
        self.clips, self.subs = [], []
        self.frames = 0
        self.proc = None
        self.turns = open(os.path.join(out_dir, "turns.jsonl"), "a",
                          encoding="utf-8")
        if not video:
            pass            # a VRM session records its states (flow.vrm)
        elif shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe",
                 "-c:v", "mjpeg", "-framerate", f"{self.fps:g}", "-i", "-",
                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt",
                 "yuv420p", os.path.join(out_dir, "video.mp4")],
                stdin=subprocess.PIPE)
        else:
            log("live: no ffmpeg; --record keeps audio, subtitles and turns "
                "only")

    def frame(self, f, jpeg):
        if self.proc is None:
            return
        try:
            # Any frames missing before this one (skipped, dropped) are
            # this one again.
            for _ in range(max(f + 1 - self.frames, 0)):
                self.proc.stdin.write(jpeg)
            self.frames = max(self.frames, f + 1)
        except (BrokenPipeError, OSError):
            self.proc = None

    def audio(self, t, x, sr):
        with self.lock:
            self.clips.append((t, np.asarray(x, np.float32), sr))

    def sub(self, s):
        with self.lock:
            self.subs.append(s)

    def turn(self, summary):
        with self.lock:          # the memory thread writes here too
            self.turns.write(json.dumps(summary, ensure_ascii=False) + "\n")
            self.turns.flush()

    def close(self):
        self.turns.close()
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        with self.lock:
            clips, subs = list(self.clips), list(self.subs)
        wav = os.path.join(self.dir, "speech.wav")
        if clips:
            sr = clips[0][2]
            end = max(t + len(x) / s for t, x, s in clips)
            n = int(math.ceil(max(end, self.frames / self.fps) * sr))
            mix = np.zeros(n, np.float32)
            for t, x, s in clips:
                y = sp._resample(x, s, sr)
                i = int(round(t * sr))
                mix[i:i + len(y)] += y[:max(n - i, 0)]
            sp.write_wav(wav, np.clip(mix, -1, 1), sr)
        srt = os.path.join(self.dir, "subs.srt")
        with open(srt, "w", encoding="utf-8") as fh:
            fh.write(dr.srt(sorted(subs, key=lambda s: s["start"])))
        vid = os.path.join(self.dir, "video.mp4")
        cmd = dr.mux_cmd(vid, wav, srt, os.path.join(self.dir,
                                                     "session.mp4"))
        if cmd and os.path.exists(vid) and shutil.which("ffmpeg"):
            r = subprocess.run(cmd, capture_output=True)
            self.log("live: session -> " + cmd[-1] if r.returncode == 0 else
                     "live: ffmpeg could not mux the session: "
                     + r.stderr.decode("utf-8", "replace")[-300:])


# ------------------------------------------------------------------ app

class App:
    """The pieces the server talks to."""

    def __init__(self, clock, hub, fps, performer=None, loop=None):
        self.clock, self.hub, self.fps = clock, hub, fps
        self.performer, self.loop = performer, loop
        # --avatar vrm / both: {"path": the .vrm, "config": the page's
        # avatar.json}; None serves the Live2D page.
        self.vrm = None

    def say(self, text):
        return self.performer.say(text)

    def state(self):
        out = {"t": self.clock.now(), "fps": self.fps}
        if self.loop is not None:
            out.update(frame=self.loop.f,
                       rendered=getattr(self.loop, "rendered", self.loop.f),
                       skipped=getattr(self.loop, "skipped", 0))
        if self.performer is not None:
            out.update(turns=self.performer.turns,
                       busy=self.performer.busy)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dr.add_args(ap)
    ap.set_defaults(backend="deepseek", ollama_timeout=120.0,
                    ollama_keep_alive="30m")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to serve on (127.0.0.1: reach it through "
                         "ssh -L 8765:127.0.0.1:8765 server)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="record the session: video.mp4, speech.wav, "
                         "subs.srt, turns.jsonl, session.mp4")
    ap.add_argument("--say", default=None, help="a first line, said at start")
    ap.add_argument("--lead", type=float, default=0.25,
                    help="seconds between a sentence being ready and its "
                         "audio starting: time to merge its tokens before "
                         "the render reaches them and to get the audio to "
                         "the page")
    ap.add_argument("--gap", type=float, default=0.12,
                    help="silence between sentences (each TTS clip already "
                         "has its own lead-in and tail)")
    ap.add_argument("--history-turns", type=int, default=5, metavar="K",
                    help="send the model only the last K user / assistant "
                         "exchanges (and the current line): the student was "
                         "trained on sessions of 6 turns and copies its own "
                         "replies past that [5]; 0 keeps the whole "
                         "conversation")
    ap.add_argument("--history-markers", type=int, default=HISTORY_MARKERS,
                    metavar="N",
                    help="older replies are sent as spoken text only; the "
                         "last N keep their [pose] markers, so she does not "
                         "copy her own old beats (they are dropped N + "
                         "--trim-chunk replies back, a chunk at a time, so "
                         f"the sent prefix holds) [{HISTORY_MARKERS}]; 0 "
                         "keeps every marker")
    ap.add_argument("--trim-chunk", type=int, default=TRIM_CHUNK,
                    metavar="C",
                    help="past --history-turns (or the context budget) drop "
                         "C exchanges at once, down to cap - C + 1, so the "
                         "prompt prefix stays the same for the turns "
                         "between trims and Ollama's prefix cache holds "
                         "(C is at most a third of --history-turns; 1: one "
                         f"exchange per turn, as before) [{TRIM_CHUNK}]")
    ap.add_argument("--glance", default="auto", choices=("auto", "off"),
                    help="auto: when the line puts something near her "
                         "(\"at your right foot\", \"behind you\", 足元に, "
                         "你脚边) without asking her to look, her reply "
                         "opens with a 0.75 s glance there (body mode) "
                         "[auto]")
    ap.add_argument("--repair", default=rp.DEFAULT_MODE, choices=rp.MODES,
                    help="auto: each sentence's markers go through the "
                         "checkers (flow.checks) before they are planned "
                         "and a checkable failure is fixed in that sentence "
                         "(a look to the wrong side flipped, \"look at me\" "
                         "centred, [frame full] before a dance, a wave that "
                         "was asked for added), logged as \"repair: ...\", "
                         "recorded in turns.jsonl and, with --record, "
                         "written as a DPO pair to repair_pairs.jsonl; "
                         "log: check and log, change nothing; off: neither "
                         f"[{rp.DEFAULT_MODE}]; body mode")
    ap.add_argument("--reminders", default="off", choices=("auto", "off"),
                    help="auto: the rules the turn needs (flow.rules: a "
                         "look to a side, \"look at me\", a wave or a dance, "
                         "a place near her, a corrected word, a game, quiet "
                         "chat, a switch of language; at most 3, with the "
                         "rules they need) are put in a \"(reminders: ...)\" "
                         "line behind the memory notes, on the last line of "
                         "the latest user message (body mode); off: nothing "
                         "added [off]")
    ap.add_argument("--quiet-after", type=float, default=20.0, metavar="S",
                    help="seconds after a reply's speech ends without a "
                         "message before the character is sent "
                         f"\"{pz.QUIET_CHAT}\" (plus the usual camera and "
                         "reply-language lines) and carries on talking by "
                         "itself [20]; 0 is never")
    ap.add_argument("--quiet-max", type=int, default=QUIET_MAX, metavar="N",
                    help="quiet turns in a row before a real message is "
                         f"waited for [{QUIET_MAX}]")
    ap.add_argument("--first-token-timeout", type=float,
                    default=FIRST_TOKEN_TIMEOUT, metavar="S",
                    help="seconds to wait for the backend's first token "
                         "before the try is abandoned and made once more "
                         "(twice stalled: the turn is dropped, its user "
                         f"line not kept in the history) [{FIRST_TOKEN_TIMEOUT:g}]; "
                         "0 waits for ever")
    ap.add_argument("--stall-timeout", type=float, default=STALL_TIMEOUT,
                    metavar="S",
                    help="seconds without a token mid-reply before the "
                         "reply is ended where it is (what was said stays "
                         f"in the history) [{STALL_TIMEOUT:g}]; 0 waits for ever")
    ap.add_argument("--batch-max", type=int, default=BATCH_MAX, metavar="N",
                    help="chat lines that came in while she was talking are "
                         "answered as one turn, the last N of them "
                         f"[{BATCH_MAX}]; 0 keeps them all")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the backend warm-up at start (a load-only "
                         "request, so the first turn is not the cold model "
                         "load; --warmup is the physics one)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    ap.add_argument("--renderer", default="gl", choices=("gl", "none"),
                    help="none: no GL, a flat test image (the pipeline, page "
                         "and audio still run)")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds of physics on the first pose before the "
                         "clock starts")
    ap.add_argument("--bg", default="0.12,0.12,0.14",
                    help="background colour, r,g,b in 0..1")
    mem.add_args(ap)
    a = ap.parse_args(argv)
    mem.check_args(a, ap)
    if a.avatar == "vrm":
        return vrm_live(a, ap)
    dr.check_live2d_args(a, ap)
    import importlib.util
    if importlib.util.find_spec("PIL") is None:     # frames are JPEG via PIL
        ap.error("flow.live needs Pillow for the frames: pip install Pillow")
    if a.renderer == "gl":
        _, ensure_display = fs.rt._backend()
        ensure_display(["-m", f"{__package__ or 'flow'}.live"]
                       + (argv if argv is not None else sys.argv[1:]))
    run = dr.prepare(a)
    rig = fs.EventRig(a.model_json, a.rig, a.events, a.clips, a.binding,
                      a.fps, a.probe_dir, a.idle_profile, a.no_snap,
                      a.events_pool_only, breath_gain=a.breath_gain,
                      voice_config=dr.voice_config_path(a))
    backend = dr.make_backend(a)
    warm_backend(a, backend)
    if a.tts is None:
        a.tts = "auto"           # live is for talking: speak by default
    tts = dr.make_voice(a)
    cal = getattr(a, "mouth_cal", None) or {}
    clock = Clock()
    hub = Hub(clock)
    stream = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle,
                             head_drift=fs.drift_from_args(a),
                             pose_ease=fs.ease_from_args(a),
                             baseline_scale=fs.baseline_from_args(a),
                             squint=fs.squint_from_args(a, rig, print),
                             mouth_map=cal.get("map")
                             if (a.lipsync or "auto") == "auto" else None,
                             log=print)
    recorder = Recorder(a.record, a.fps) if a.record else None
    perf = Performer(a, run, rig, stream, clock, hub, backend, tts,
                     recorder=recorder, memory=mem.from_args(a))
    vrm_side = None
    if a.avatar == "both":
        vrm_side = both_setup(a, perf, clock, hub)
    encoder = Encoder(hub, a.quality, recorder)
    stop = threading.Event()
    # SIGTERM (timeout, systemd, a closed ssh session's kill) stops like
    # Ctrl-C, so --record still writes its files.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    bg = tuple(float(x) for x in a.bg.split(",")) + (1.0,)

    def serve(be, absent=()):
        loop = RenderLoop(stream, be, clock, encoder, a.fps, absent=absent)
        perf.snap = lambda: loop.last_img
        app = App(clock, hub, a.fps, perf, loop)
        if vrm_side is not None:
            # The VRM's states on their own thread, on the same clock.
            app.vrm = vrm_side["page"]
            threading.Thread(target=vrm_side["loop"].run, args=(stop,),
                             name="vrm", daemon=True).start()
        srv = Server(app, a.host, a.port)
        srv.start()
        print(f"live: serving http://{a.host}:{srv.server_address[1]}/ "
              f"(ssh -L {srv.server_address[1]}:127.0.0.1:"
              f"{srv.server_address[1]} <server>, then open that address)")
        perf.start()
        if a.say:
            perf.say(a.say)
        try:
            loop.run(stop)
        except KeyboardInterrupt:
            print("live: stopping")
        finally:
            stop.set()
            srv.stop()
            encoder.close()
            if recorder is not None:
                recorder.close()
            if vrm_side is not None and vrm_side["log"] is not None:
                print("live: vrm states -> " + vrm_side["log"].close())

    if a.renderer == "none":
        if perf.framing is not None:
            # No GL, no body frame to place a window by: a stand-in zoom,
            # so [frame] markers still run through the pipeline.
            stream.camera = fr.Camera({"bust": (1.7, 0.0, 0.1),
                                       "full": (1.0, 0.0, 0.0)}, a.frame)
        serve(FakeBackend(a.size))
        return 0
    Live2DBackend, _ = fs.rt._backend()
    with Live2DBackend(a.model_json, size=a.size, bg=bg,
                       strip_dynamics=False, fps=a.fps) as be:
        be.calibrate()
        if getattr(a, "frame", "full") == "bust" or perf.framing is not None:
            from flow import framing
            stream.camera = framing.camera_for(
                be, framing.body_path_for(a.rig, a.model_json), a.model_json,
                getattr(a, "frame", "full"))
            if stream.camera is None and perf.framing is not None:
                perf.framing = perf.home = "full"   # no bust view
        print("backend: " + ", ".join(
            f"{k}={be._api_report[k]}" for k in ("order", "timestep", "parts")
            if k in be._api_report))
        by_id = {s.pid for s in be.params}
        absent = rig.expr_ids - by_id
        # Physics settles on the first pose before the clock starts; the
        # stream's own first frame is computed again from frame 0.
        warm = fs.MotionStream(rig, seed=a.seed, idle=False, autonomic=False,
                               log=lambda s: None)
        over, parts = warm.frame()
        for _ in range(int(a.warmup * a.fps)):
            be.render({p: v for p, v in over.items() if p not in absent},
                      parts=parts)
        clock.t0 = time.monotonic()
        serve(be, absent)
    return 0


def warm_backend(a, backend, log=print) -> float | None:
    """Load the model before the first turn (the cold load took ~9 s of the
    first reply): the backend's warmup() when it has one (Ollama: a request
    with no messages), timed and logged. A failure only warns. Returns the
    seconds, or None when skipped or failed."""
    fn = getattr(backend, "warmup", None)
    if getattr(a, "no_warmup", False) or fn is None:
        return None
    try:
        dt = float(fn())
    except Exception as exc:  # noqa: BLE001
        log(f"live: backend warm-up failed ({exc!r}); the first turn may be "
            "slow")
        return None
    log(f"live: backend warmed up in {dt:.1f}s")
    return dt


def both_setup(a, perf, clock, hub):
    """--avatar both: a VRM beside the Live2D rig. Its own stream (idle,
    blinks, breath) follows every sentence the Live2D rig plays (flow.vrm.
    Follower); the prompt stays the Live2D body's (the more constrained
    one: a target it reaches the VRM reaches too)."""
    try:
        from . import vrm as fv
    except ImportError:
        from flow import vrm as fv
    bm, bpath, rig, page = fv.live_setup(a)
    stream = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle,
                             head_drift=fs.drift_from_args(a),
                             pose_ease=fs.ease_from_args(a),
                             baseline_scale=fs.baseline_from_args(a), log=print)
    perf.followers.append(fv.Follower(a, bm, rig, stream, log=perf.emit))
    slog = (fv.StateLog(a.record, a.fps, bpath, page["path"]) if a.record
            else None)
    loop = fv.StateLoop(stream, bm, clock, hub, a.fps, record=slog)
    return {"page": page, "loop": loop, "log": slog}


def vrm_live(a, ap) -> int:
    """main() for --avatar vrm: the same turns (flow.live's Performer) on a
    VRM body (flow.vrm); the page draws the states the server streams on
    /states. No GL, no Pillow."""
    try:
        from . import vrm as fv
    except ImportError:
        from flow import vrm as fv
    if a.vrm is None and a.vrm_body is None:
        ap.error("--avatar vrm needs --vrm PATH (or --vrm-body FILE)")
    if a.body_image:
        print("live: --body-image: the server cannot draw a VRM (the browser "
              "does); text only")
        a.body_image = False
    if (a.lipsync or "auto") == "auto":
        a.lipsync = "vowels"
    bm, bpath, rig, page = fv.live_setup(a)
    run = fv.prepare(a, bm, bpath)
    backend = dr.make_backend(a)
    warm_backend(a, backend)
    if a.tts is None:
        a.tts = "auto"
    tts = dr.make_voice(a)
    clock = Clock()
    hub = Hub(clock)
    stream = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle,
                             head_drift=fs.drift_from_args(a),
                             pose_ease=fs.ease_from_args(a),
                             baseline_scale=fs.baseline_from_args(a), log=print)
    recorder = (Recorder(a.record, a.fps, video=False) if a.record
                else None)
    slog = (fv.StateLog(a.record, a.fps, bpath, page["path"]) if a.record
            else None)
    perf = Performer(a, run, rig, stream, clock, hub, backend, tts,
                     recorder=recorder, memory=mem.from_args(a))
    loop = fv.StateLoop(stream, bm, clock, hub, a.fps, record=slog)
    app = App(clock, hub, a.fps, perf, loop)
    app.vrm = page
    stop = threading.Event()
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    srv = Server(app, a.host, a.port)
    srv.start()
    print(f"live: serving http://{a.host}:{srv.server_address[1]}/ "
          f"(VRM {os.path.basename(page['path'])}; ssh -L "
          f"{srv.server_address[1]}:127.0.0.1:{srv.server_address[1]} "
          "<server>, then open that address)")
    perf.start()
    if a.say:
        perf.say(a.say)
    try:
        loop.run(stop)
    except KeyboardInterrupt:
        print("live: stopping")
    finally:
        stop.set()
        srv.stop()
        if recorder is not None:
            recorder.close()
        if slog is not None:
            print("live: vrm states -> " + slog.close() + " (render: python "
                  f"-m flow.vrmrender --run {a.record})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
