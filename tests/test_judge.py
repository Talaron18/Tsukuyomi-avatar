"""
flow.judge: the keyframe walk (marker order, state carried from one marker
to the next, framing switches, the start state), the contact sheet, the
judgment parser, retry and cache, the call budget, and the fake judge, all
with the fake renderer: no GL, no network.
"""

from __future__ import annotations

import io
import json
import os

import pytest

from flow import body as fb
from flow import judge as jg
from flow import moves as mv
from flow import pose as pz
from tests.test_dataset import ROOT
from tests.test_dataexport import _run

HIYORI = os.path.join(ROOT, "rigs", "Hiyori.body.json")


@pytest.fixture(scope="module")
def bm():
    return fb.load(HIYORI)


def yaw_param(bm):
    return jg.carry_roles(bm)["head_x"][0]


def pose(**kw):
    kw = {k.replace("_", "."): v for k, v in kw.items()}
    return "[pose " + json.dumps(kw) + "]"


# --------------------------------------------------------------- the walk

def test_markers_come_in_order_with_their_words(bm):
    raw = ("Hi there! " + pose(brow_L_lift=0.5, brow_R_lift=0.5)
           + " Nice to see you all. " + pose(head_yaw=-15, hold=2)
           + " Look to my right. [move wave] Bye bye now")
    w = jg.walk(bm, raw, "full")
    kinds = [(k["kind"], k["n"]) for k in w["keys"]]
    assert kinds == [("rest", 0), ("pose", 1), ("pose", 2), ("move", 3)]
    assert w["keys"][1]["words"].startswith("Nice to see you all")
    assert w["keys"][2]["words"].startswith("Look to my right")
    assert w["keys"][3]["name"] == "wave"
    assert w["spoken"].startswith("Hi there! Nice")
    assert w["keys"][1]["targets"] == {"brow.L.lift": 0.5,
                                       "brow.R.lift": 0.5}
    # the first caption line: number and camera
    assert w["keys"][2]["caption"][0] == "2 full"
    assert w["keys"][1]["t"] < w["keys"][2]["t"] < w["keys"][3]["t"]


def test_head_and_face_carry_to_the_next_marker(bm):
    yaw = yaw_param(bm)
    base = bm.default_values()
    raw = (pose(head_yaw=-15, hold=2) + " Over here. "
           + pose(brow_L_lift=0.6, brow_R_lift=0.6) + " Hehe, okay.")
    k1, k2 = jg.walk(bm, raw, "full")["keys"][1:3]
    assert k1["params"][yaw] != base[yaw]
    # the head stays where the first marker put it under the second's face
    assert k2["params"][yaw] == pytest.approx(k1["params"][yaw], abs=0.5)
    brow = bm.solve({"brow.L.lift": 0.6})["params"]
    p = next(iter(brow))
    assert abs(k2["params"][p] - base[p]) > 0.05
    # a short reaction does not carry: a yaw of 4 relaxes before the next
    raw = pose(head_yaw=4, hold=0.3) + " Yes. " + pose(
        brow_L_lift=0.5) + " And then. " + pose(brow_R_lift=0.4) + " ok"
    ks = jg.walk(bm, raw, "full")["keys"]
    assert abs(ks[3]["params"][yaw] - base[yaw]) < abs(
        ks[1]["params"][yaw] - base[yaw]) + 1e-6


def test_the_turn_starts_in_the_bodys_current_state(bm):
    yaw = yaw_param(bm)
    line = ("Hello\n(your body now: head.yaw -12, hand.L (0.95, -2.00), "
            "hand.R (-0.78, -2.09))\n(camera: bust, head and chest; hands "
            "below y -1.00 are out of frame)\n(reply in: English)")
    now = jg.now_targets(line)
    assert now == {"head.yaw": -12.0}
    assert jg.now_targets("(your body now: at rest; hand.L (0.95, -2.00), "
                          "hand.R (-0.78, -2.09))") == {}
    assert jg.home_of({"user_content": line}) == "bust"
    assert jg.home_of({"user_content": "x", "framing": {"start": "full"}}) \
        == "full"
    w = jg.walk(bm, pose(brow_L_lift=0.5) + " Hi.", "bust", now)
    rest, k1 = w["keys"][:2]
    assert abs(rest["params"][yaw] - bm.default_values()[yaw]) > 1.0
    assert k1["params"][yaw] == pytest.approx(rest["params"][yaw], abs=0.5)
    assert w["keys"][0]["caption"] == ["rest (start)"]


def test_framing_home_marker_and_the_camera_following(bm):
    raw = (pose(brow_L_lift=0.5) + " Hi. [frame full] "
           + pose(head_tilt=6, hold=1) + " Now the whole me. "
           + "[frame bust] " + pose(head_yaw=8, hold=1) + " And back.")
    keys = jg.walk(bm, raw, "bust")["keys"]
    assert [k["view"] for k in keys] == ["bust", "bust", "full", "bust"]
    # started in full: the first still is full
    assert jg.walk(bm, raw, "full")["keys"][1]["view"] == "full"
    # a [frame] marker that nothing follows gets a still of its own
    keys = jg.walk(bm, pose(head_tilt=5) + " Hi. [frame full]", "bust")["keys"]
    assert [k["kind"] for k in keys] == ["rest", "pose", "frame"]
    assert keys[2]["view"] == "full" and keys[1]["view"] == "bust"
    # the camera follows a whole-body move in bust, as the runtime's does
    if not mv.caps(bm).get("twirl", {}).get("ok"):
        pytest.skip("Hiyori cannot twirl here")
    raw = "Watch! [move twirl] Wheee."
    k = jg.walk(bm, raw, "bust")["keys"][1]
    assert k["kind"] == "move" and k["view"] == "full" and k["followed"]
    k = jg.walk(bm, raw, "bust", follow=False)["keys"][1]
    assert k["view"] == "bust" and not k["followed"]
    # a hand move does not follow
    k = jg.walk(bm, "Hi! [move wave] bye", "bust")["keys"][1]
    assert k["view"] == "bust"


def test_a_move_is_taken_at_its_extreme(bm):
    raw = "Hi! [move wave] bye bye"
    k = jg.walk(bm, raw, "full")["keys"][1]
    rest = bm.default_values()
    arms = pz.arm_params(bm)
    moved = [p for p in arms if abs(k["params"].get(p, rest[p]) - rest[p])
             > 0.05]
    assert moved, "the wave's most extreme still has a raised arm"


def test_every_part_of_the_rest_drawing_is_set(bm):
    keys = jg.walk(bm, "Hi! [move wave] bye", "full")["keys"]
    want = bm.variants[bm.match_variant(bm.default_values())]["parts"]
    for k in keys:
        # a rest, a wave: no still leaves an arm drawing unset
        assert set(want) <= set(k["parts"]), k["caption"]
    assert {p: keys[0]["parts"][p] for p in want} == want


def test_a_body_without_the_variable_drops_the_marker(bm):
    keys = jg.walk(bm, '[pose {"tail.wag": 1}] Hi.', "full")["keys"]
    assert [k["kind"] for k in keys] == ["rest"]


# -------------------------------------------------------- the contact sheet

def test_sheet_composition_and_tile_choice(bm):
    from PIL import Image
    r = jg.FakeRenderer()
    raw = " ".join(pose(brow_L_lift=0.1 * i, brow_R_lift=0.1 * i) + f" w{i}."
                   for i in range(1, 9)) + " [move wave] Bye now"
    sh = jg.render_sheet(bm, r, raw, "bust", max_tiles=5, tile=128)
    im = Image.open(io.BytesIO(sh.png))
    assert im.size[0] == 5 * 128 and im.size[1] > 128
    assert sh.n_keys == 9 and len(sh.keys) == 5
    # the start state first, the move (priority) kept, order kept
    assert sh.keys[0]["kind"] == "rest"
    assert any(k["kind"] == "move" for k in sh.keys)
    ns = [k["n"] for k in sh.keys[1:]]
    assert ns == sorted(ns)
    assert len(r.calls) == 5
    # the shown stills ride along in the PNG for the fake judge
    assert json.loads(im.info["tsk.keys"])[0]["kind"] == "rest"
    # the camera is set when it changes: bust throughout is one set_view
    assert r.be.views == [r.views["bust"]]


def test_sheet_sets_the_view_when_the_framing_switches(bm):
    r = jg.FakeRenderer()
    raw = pose(head_tilt=5) + " Hi. [frame full] " + pose(
        head_tilt=-5, hold=1) + " Whole me."
    jg.render_sheet(bm, r, raw, "bust")
    assert r.be.views == [r.views["bust"], r.views["full"]]


def test_priority_prefers_moves_hands_and_looks():
    mk = lambda kind, t: {"kind": kind, "targets": t, "n": 1}  # noqa: E731
    assert jg.priority(mk("move", {})) == 3
    assert jg.priority(mk("pose", {"hand.R.y": 0.2})) == 3
    assert jg.priority(mk("pose", {"head.yaw": -15})) == 2
    assert jg.priority(mk("pose", {"head.yaw": -3})) == 1
    assert jg.priority(mk("pose", {"brow.L.lift": 0.5})) == 1


# ------------------------------------------------------------ the judgment

@pytest.mark.parametrize("text,want", [
    ('{"request_done": 2, "visible": 1, "expression_fit": 2, "natural": 4, '
     '"issues": []}', (2, 1, 2, 4)),
    ('```json\n{"request_done": null, "visible": "NA", "expression_fit": 1, '
     '"natural": 3, "issues": ["flat"]}\n```', (None, None, 1, 3)),
    ('<think>hmm {"natural": 1}</think>Sure: {"request_done": "n/a", '
     '"visible": 2, "expression_fit": 2, "natural": 9, "issues": "x"}',
     (None, 2, 2, 5)),                                  # clamped, a string
    ('{"request_done": 7, "expression_fit": -3, "natural": 0}',
     (2, None, 0, 1)),
])
def test_parse_judgment_is_lenient(text, want):
    got = jg.parse_judgment(text)
    assert tuple(got[f] for f in jg.FIELDS) == want
    assert isinstance(got["issues"], list)


@pytest.mark.parametrize("text", ["", "I cannot tell.", "{broken",
                                  '{"request_done": 2}',
                                  '{"expression_fit": "great", "natural": 3}',
                                  '[1, 2]'])
def test_parse_judgment_rejects_the_unusable(text):
    assert jg.parse_judgment(text) is None


def _sheet(bm, raw, home="bust"):
    return jg.render_sheet(bm, jg.FakeRenderer(), raw, home)


def test_retry_once_on_bad_json_and_count(bm, tmp_path):
    sh = _sheet(bm, pose(head_tilt=5) + " Hi.")
    good = json.dumps({"request_done": None, "visible": None,
                       "expression_fit": 2, "natural": 4, "issues": []})
    be = jg.FakeJudgeBackend(raw=lambda n: "no idea" if n == 1 else good)
    j = jg.make_judge("fake", fake=be)
    got = j.score(sh, "Hello", "Hi.")
    assert got["scores"]["natural"] == 4
    assert (j.calls, j.retries, j.bad) == (2, 1, 0)
    # never parses: two calls, no scores, counted bad, nothing cached
    be = jg.FakeJudgeBackend(raw="still no")
    j = jg.make_judge("fake", fake=be, cache=str(tmp_path))
    assert j.score(sh, "Hello", "Hi.")["scores"] is None
    assert (j.calls, j.bad) == (2, 1)
    assert not [f for _d, _s, fs in os.walk(tmp_path) for f in fs]


def test_cache_by_sheet_prompt_and_model(bm, tmp_path):
    sh = _sheet(bm, pose(head_tilt=5) + " Hi.")
    be = jg.FakeJudgeBackend()
    j = jg.make_judge("fake", fake=be, cache=str(tmp_path))
    a = j.score(sh, "Hello", "Hi.")
    b = j.score(sh, "Hello", "Hi.")
    assert (j.calls, j.cached) == (1, 1) and not a["cached"] and b["cached"]
    assert a["scores"] == b["scores"]
    # another user line, another sheet or another model is another key
    j.score(sh, "Hello again", "Hi.")
    j.score(_sheet(bm, pose(head_tilt=-9) + " Hi."), "Hello", "Hi.")
    assert j.calls == 3
    j2 = jg.make_judge("fake", "other-model", fake=be, cache=str(tmp_path))
    j2.score(sh, "Hello", "Hi.")
    assert j2.calls == 1
    # a new run on the same cache costs nothing
    j3 = jg.make_judge("fake", fake=jg.FakeJudgeBackend(),
                       cache=str(tmp_path))
    j3.score(sh, "Hello", "Hi.")
    assert (j3.calls, j3.cached) == (0, 1)
    assert "from cache" in j3.report() and "tokens" in j3.report()


def test_call_budget(bm):
    sh = _sheet(bm, pose(head_tilt=5) + " Hi.")
    j = jg.make_judge("fake", fake=jg.FakeJudgeBackend(), max_calls=2)
    j.score(sh, "a", "x")
    j.score(sh, "b", "x")
    with pytest.raises(jg.BudgetExceeded):
        j.score(sh, "c", "x")
    assert j.calls == 2


def test_the_request_carries_the_image_as_a_data_url(bm):
    sh = _sheet(bm, pose(head_tilt=5) + " Hi.")
    seen = {}

    class Spy(jg.FakeJudgeBackend):
        def stream(self, system, messages):
            seen["system"], seen["msg"] = system, messages
            return super().stream(system, messages)
    j = jg.make_judge("fake", fake=Spy())
    j.score(sh, "Look to your right!\n(camera: bust)", "Sure.",
            {"hand_moves": {"bust": 0, "full": 2}, "face": 12,
             "reach": {"all": 0.4}}, "Hiyori")
    c = seen["msg"][0]["content"]
    assert c[0]["image_url"]["url"].startswith("data:image/png;base64,")
    text = c[1]["text"]
    assert "Look to your right!" in text and "(camera: bust)" in text
    assert "Sure." in text and "hand moves a reply: bust 0, full 2" in text
    assert "tile 1: start state" in text and "tile 2:" in text
    assert "request_done" in seen["system"] and "JSON" in seen["system"]
    assert "VIEWER's LEFT" in seen["system"]


def test_fake_judge_reads_the_look_side_and_the_hand(bm):
    j = jg.make_judge("fake")
    user = "Look to your right!"
    right = _sheet(bm, "Sure! " + pose(head_yaw=-15, gaze_x=-0.5, hold=2)
                   + " There.")
    left = _sheet(bm, "Sure! " + pose(head_yaw=15, gaze_x=0.5, hold=2)
                  + " There.")
    assert j.score(right, user, "Sure!")["scores"]["request_done"] == 2
    assert j.score(left, user, "Sure!")["scores"]["request_done"] == 0
    # a hand out of the bust picture is not visible; in full it is
    hand = "Hi! " + pose(hand_R_y=-1.6, hand_R_x=0.5) + " there"
    assert j.score(_sheet(bm, hand, "bust"), "Wave at me", "Hi")[
        "scores"]["visible"] == 0
    assert j.score(_sheet(bm, hand, "full"), "Wave at me", "Hi")[
        "scores"]["visible"] == 2
    assert j.score(right, "How are you?", "x")["scores"]["request_done"] \
        is None


def test_backends_and_defaults():
    assert set(jg.JUDGE_BACKENDS) == {"siliconflow", "vllm", "anthropic",
                                      "fake"}
    j = jg.make_judge("vllm", api_base="http://127.0.0.1:1/v1")
    be = j.factory()
    assert be.sampling == {"temperature": 0}      # not the student's
    assert j.fmt == "siliconflow"                 # image_url data URLs
    assert "VL" in j.model
    assert jg.make_judge("siliconflow").model == jg.JUDGE_MODELS["siliconflow"]
    with pytest.raises(SystemExit):
        jg.make_judge("nope")


# ------------------------------------------------------------------- the cli

def test_render_cli_with_the_fake_renderer(tmp_path, capsys):
    out, recs = _run(tmp_path, "Hi! " + pose(head_yaw=-15, hold=1)
                     + " ok [move wave] bye", avatars="Hiyori", sessions=1,
                     turns=2, prompt="beats2")
    rid = sorted(r["id"] for r in recs)[1]
    png = str(tmp_path / "s.png")
    rc = jg.main(["render", "--data", out, "--id", rid, "--renderer", "fake",
                  "--out", png])
    assert rc == 0 and os.path.getsize(png) > 1000
    assert "tiles of" in capsys.readouterr().out
    # a reply given on the command line replaces the record's
    rc = jg.main(["render", "--data", out, "--id", rid, "--renderer", "fake",
                  "--reply", "Just words.", "--out", png])
    assert rc == 0
    assert "1 tiles of 0" in capsys.readouterr().out
    rc = jg.main(["score", "--data", out, "--id", rid, "--renderer", "fake",
                  "--judge-backend", "fake"])
    assert rc == 0
    assert '"natural"' in capsys.readouterr().out


def test_the_judge_is_blind_to_the_targets_by_default(bm, monkeypatch):
    raw = (pose(head_yaw=-15, gaze_x=-0.6) + " Over there. [move wave] Hi!")
    sh = jg.render_sheet(bm, jg.FakeRenderer(), raw, "bust")
    keys = sh.keys
    blind = jg.prompt_text("Look to your right!", "Over there. Hi!", None,
                           keys)
    assert "head.yaw" not in blind and "wave" not in blind
    assert "before \"Over there" in blind
    seen = jg.prompt_text("Look to your right!", "Over there. Hi!", None,
                          keys, show_targets=True)
    assert "head.yaw" in seen and "move wave" in seen
    # the captions drawn under the tiles follow the same switch
    w = jg.walk(bm, raw, "bust", None, 0, True)
    lines = [ln for k in w["keys"] for ln in k.get("caption", [])]
    assert not any("head.yaw" in ln or "wave" in ln for ln in lines)
    assert "targets" not in jg.RUBRIC.split("Sides:")[0]


def test_failed_calls_are_counted_apart_from_unparsed_answers(bm, capsys):
    sh = _sheet(bm, pose(head_tilt=5) + " Hi.")

    class Down:
        usage, stop_reason = {}, None

        def stream(self, system, messages):
            raise RuntimeError("403 from siliconflow: " + "x" * 400)
            yield ""

    j = jg.Judge(lambda: Down(), "m", "siliconflow")
    with pytest.raises(RuntimeError):
        j.score(sh, "Hello", "Hi.")
    assert (j.calls, j.failed, j.bad) == (1, 1, 0)
    err = capsys.readouterr().err
    assert "403 from siliconflow" in err and len(err) < 420
    assert "1 failed, 0 unparsed" in j.report() and "403" in j.report()
    # an empty 200 is shown too, and is an unparsed answer, not a failure
    class Empty(Down):
        def stream(self, system, messages):
            self.usage = {"prompt_tokens": 900, "completion_tokens": 0}
            self.stop_reason = "length"
            yield ""

    j = jg.Judge(lambda: Empty(), "m", "siliconflow")
    assert j.score(sh, "Hello", "Hi.")["scores"] is None
    assert (j.calls, j.failed, j.bad) == (2, 0, 1)
    assert "empty answer (stop length" in capsys.readouterr().err


def test_sheet_defaults_keep_the_image_cost():
    assert (jg.TILE, jg.MAX_TILES) == (320, 5)
    # about the pixels of the 6 x 256 sheet of before
    assert 0.9 <= (jg.TILE ** 2 * jg.MAX_TILES) / (256 ** 2 * 6) <= 1.4
