"""
flow.dpo: the prompts equal the SFT export's, the minimal edits (each kind
breaks exactly one thing and the checker sees it), the score, the contradiction rule,
the choice of a pair, calibration and the whole run with the fake renderer and the fake
judge. No GL, no network, no model.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from flow import checks as ck
from flow import dataexport as dx
from flow import dataset as ds
from flow import dpo
from flow import judge as jg
from flow import pose as pz
from tests.test_dataset import CharLLM, UserLLM

BODIES = ck.Bodies(budgets={
    "Hiyori": {"hand_moves": {"bust": 1, "full": 2},
               "frames": ["bust", "full"]}})


def turns_of(out, *a, **kw):
    """The data is live2d-lite (no rig here): keep_lite, as the export's."""
    kw.setdefault("keep_lite", True)
    return dpo.load_turns(out, *a, **kw)


def pose(**kw):
    kw = {k.replace("_", "."): v for k, v in kw.items()}
    return "[pose " + json.dumps(kw) + "]"


def rec(raw, user="Hello!"):
    return ck.make_rec(raw, user, "Hiyori", "s", 1)


def spoken(raw):
    return dx.spoken_of(raw)


def markers(raw):
    return [pz.tag_text(t) for _a, _b, t in dx.marker_spans(raw)]


# ------------------------------------------------------------------ prompts

REPLY = ("Hehe! " + pose(brow_L_lift=0.5, brow_R_lift=0.5, head_tilt=4)
         + " Sure, here. " + pose(head_yaw=-15, gaze_x=-0.5, hold=2)
         + " Yes! [move wave] Bye now.")


# One reply a turn (the mock user asks to wave on turn 2): a reply the same
# as the turn before would fail template_repeat.
WAVE_REPLY = ("Sure! " + pose(brow_L_lift=0.4, brow_R_lift=0.4)
              + " Waving. " + pose(hand_R_y=0.4, hand_R_x=0.5, dur=0.4)
              + " [move wave] Hi hi!")
THIRD = "Oh? " + pose(brow_L_lift=0.3, head_tilt=4) + " Okay then."


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("dpo") / "run")

    def make(role, av):
        return CharLLM([REPLY, WAVE_REPLY, THIRD]) if role == "character" \
            else UserLLM()
    opts = ds.parse_opts(["gen", "--avatars", "Hiyori,seed-san",
                          "--live2d-lite", "--sessions", "4", "--turns", "3",
                          "--out", out, "--personas", "cheerful",
                          "--teacher-prompt", "beats2"])
    ds.generate(opts, backend_factory=make, log=lambda s: None)
    return out, ds.iter_records(out)


@pytest.mark.parametrize("student", ["teacher", "short"])
def test_prompt_and_teacher_reply_are_the_sft_exports(data, student):
    out, recs = data
    turns = turns_of(out, student=student)
    assert turns
    plan = dx.prepare(out, recs, keep_lite=True)
    ex = dx.sft_examples(out, recs, plan["notes"], "turn", True,
                         {**plan["fix"], **plan["mfix"]}, None, student)
    by = {e["meta"]["id"]: e["messages"] for e in ex}
    assert {t["id"] for t in turns} == set(by)
    for t in turns:
        msgs = by[t["id"]]
        assert t["prompt"] == [{"role": m["role"], "content": m["content"]}
                               for m in msgs[:-1]]
        assert t["teacher"] == msgs[-1]["content"]
        assert t["prompt"][0]["role"] == "system"
        assert t["prompt"][-1]["role"] == "user"
        assert all(set(m) == {"role", "content"} for m in t["prompt"])
    # history is part of the prompt on a later turn
    last = next(t for t in turns if t["id"].endswith("/t03"))
    assert len(last["prompt"]) == 1 + 2 * 2 + 1
    if student == "short":
        full = turns_of(out)
        assert len(turns[0]["prompt"][0]["content"]) < len(
            full[0]["prompt"][0]["content"])


def test_lite_avatars_are_not_pairs_unless_the_export_keeps_them(data):
    out, _recs = data
    # dataexport sft's default: Hiyori here has no rig (live2d-lite)
    plain = {t["rec"]["avatar"]["name"] for t in dpo.load_turns(out)}
    assert plain == {"seed-san"}
    assert {t["rec"]["avatar"]["name"] for t in turns_of(out)} == {
        "seed-san", "Hiyori"}


def test_sessions_and_avatars_are_selected(data, tmp_path):
    out, recs = data
    sid = sorted({r["session"] for r in recs})[0]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"eval_sessions": [sid],
                                 "train_sessions": ["x"]}))
    turns = turns_of(out, dpo.split_sessions(str(split)))
    assert {t["rec"]["session"] for t in turns} == {sid}
    assert dpo.split_sessions(str(split), "train") == {"x"}
    assert {t["rec"]["avatar"]["name"]
            for t in turns_of(out, avatars=["seed-san"])} == {"seed-san"}


# ------------------------------------------------------------ minimal edits

LOOK = ("Sure! " + pose(head_yaw=-15, gaze_x=-0.5, hold=2) + " Over there.")


def test_look_flip_flips_the_sign_and_nothing_else():
    user = "Look to your right!"
    bad = dpo.flip_look(LOOK, user)
    assert bad == LOOK.replace("-15", "15").replace("-0.5", "0.5")
    assert spoken(bad) == spoken(LOOK)
    v = ck.CHECKS["look_sign"]
    assert v([rec(LOOK, user)], BODIES)[0].popitem()[1] == ck.PASS
    got, _x, why = v([rec(bad, user)], BODIES)
    assert list(got.values()) == [ck.FAIL] and list(why.values()) == [
        "wrong_sign"]
    # left is the other way round
    assert dpo.flip_look(bad, "Look to your left!") == LOOK
    # no pair when no look was asked, the reply got it wrong already, or
    # nothing turns
    assert dpo.flip_look(LOOK, "How are you?") is None
    assert dpo.flip_look(LOOK, "Look to your left!") is None
    assert dpo.flip_look("Sure " + pose(brow_L_lift=0.5), user) is None
    # only the first turning marker changes
    two = LOOK + " " + pose(head_yaw=-20) + " and again"
    new = dpo.flip_look(two, user)
    assert markers(new)[1] == markers(two)[1]


HAND = ("Hi! " + pose(brow_L_lift=0.5, hand_R_y=0.4, hand_R_x=0.5, dur=0.4)
        + " here. [move wave] bye " + pose(hand_L_y=0.3) + " now")


def test_hand_drop_removes_the_hand_and_nothing_else():
    user = "Can you wave to me?"
    bad = dpo.drop_hand(HAND, user)
    assert spoken(bad) == spoken(HAND)
    assert "move" not in bad and "hand" not in bad
    # the face marker stays with its own keys (dur too), the hand-only goes
    assert markers(bad) == [pz.tag_text(
        {"tag": "pose", "arg": {"targets": {"brow.L.lift": 0.5},
                                "dur": 0.4}})]
    assert ck.has_hand(rec(HAND)) and not ck.has_hand(rec(bad))
    for name in ("wave_obeyed", "body_request_obeyed"):
        a = ck.CHECKS[name]([rec(HAND, user)], BODIES)[0]
        b = ck.CHECKS[name]([rec(bad, user)], BODIES)[0]
        assert list(a.values()) == [ck.PASS] and list(b.values()) == [ck.FAIL]
    # not applicable: no request, no hands, nothing to remove
    assert dpo.drop_hand(HAND, "How are you?") is None
    assert dpo.drop_hand(HAND, user, has_hands=False) is None
    assert dpo.drop_hand("Hi! " + pose(brow_L_lift=0.5), user) is None


BEAT = ("Oh! " + pose(brow_L_lift=0.3, eye_L_smile=0.5, head_tilt=4)
        + " Hi there. " + pose(brow_R_lift=0.2, head_yaw=3) + " How are you?")


def test_template_open_replaces_the_first_beat_only():
    bad = dpo.template_open(BEAT, "Hi!")
    assert ck.is_q8r2_opening(rec(bad)) and not ck.is_q8r2_opening(rec(BEAT))
    assert spoken(bad) == spoken(BEAT)
    assert markers(bad)[1:] == markers(BEAT)[1:]
    first = dx.marker_spans(bad)[0][2]["arg"]
    assert first == {"targets": {k: float(v) for k, v in dpo.TEMPLATE.items()
                                 if k not in ("dur", "hold")},
                     "dur": 0.2, "hold": 0.4}
    v = ck.CHECKS["opening"]
    assert v([rec(bad)], BODIES)[0].popitem()[1] == ck.FAIL
    assert v([rec(BEAT)], BODIES)[0].popitem()[1] == ck.PASS
    assert dpo.template_open(bad, "Hi!") is None          # already so
    assert dpo.template_open("Hi " + pose(hand_R_y=0.4), "Hi") is None
    assert dpo.template_open(BEAT, "Hi!", offered=["head.pitch"]) is None
    assert dpo.template_open(BEAT, "Hi!", offered=list(dpo.TEMPLATE)) == bad
    # the look the user asked for is not replaced
    turn = "Sure " + pose(brow_L_lift=0.3, head_yaw=-15) + " there."
    assert dpo.template_open(turn, "Look to your right") is None


def test_beat_dup_repeats_a_beat_and_nothing_else():
    bad = dpo.beat_dup(BEAT)
    assert spoken(bad) == spoken(BEAT)
    m, b = markers(BEAT), markers(bad)
    assert len(b) == len(m) + 1 and b == [m[0], m[0], m[1]]
    assert ck.signature(rec(bad)) != ck.signature(rec(BEAT))
    # one beat: the copy goes after the next sentence
    one = "Oh! " + pose(brow_L_lift=0.3) + " Hi there. How are you?"
    b = markers(dpo.beat_dup(one))
    assert b == [markers(one)[0]] * 2
    assert dpo.beat_dup(one).index("How") > dpo.beat_dup(one).rindex("[pose")
    assert dpo.beat_dup("No markers at all.") is None
    assert dpo.beat_dup("Hi " + pose(hand_R_y=0.4)) is None


def test_synthetic_pairs_on_the_data(data):
    out, _recs = data
    turns = turns_of(out)
    sp = dpo.synthetic_pairs(turns, BODIES, 2, seed=1)
    kinds = {k for k, *_ in sp}
    assert {"template_open", "beat_dup"} <= kinds
    for kind, t, intact, broken in sp:
        assert intact == t["teacher"] and broken != intact
        assert spoken(broken) == spoken(intact)
    rows, metas = dpo.synthetic_rows(sp, BODIES)
    assert len(rows) == len(metas) == len(sp)
    assert all(m["kind"].startswith("synthetic:") for m in metas)
    by = {m["kind"]: m for m in metas}
    if "synthetic:template_open" in by:
        assert by["synthetic:template_open"]["rejected"]["verdicts"][
            "opening"] == ck.FAIL


WAVE = ("Hi! " + pose(brow_L_lift=0.5, head_tilt=3) + " Bye! [move wave] See you.")


def test_wave_frame_inserts_a_full_frame_before_the_wave_only():
    bad = dpo.wave_frame(WAVE)
    assert bad == WAVE.replace("[move wave]", "[frame full] [move wave]")
    assert spoken(bad) == spoken(WAVE)
    assert markers(bad) == [markers(WAVE)[0], "[frame full]",
                            markers(WAVE)[1]]
    v = ck.CHECKS["frame_before_wave"]
    assert v([rec(WAVE)], BODIES)[0].popitem()[1] == ck.PASS
    assert v([rec(bad)], BODIES)[0].popitem()[1] == ck.FAIL
    # nothing to do: no wave, or a full frame already ahead of it
    assert dpo.wave_frame("Hi " + pose(head_tilt=3)) is None
    assert dpo.wave_frame(bad) is None
    assert dpo.synthetic_pair("wave_frame", WAVE, "Bye") == bad
    # the marker glued to the word before it gets a space
    glued = dpo.wave_frame("Bye[move wave] now")
    assert glued == "Bye [frame full] [move wave] now"


TOP = {"brow.L.lift": 0.5, "brow.R.lift": 0.5, "head.tilt": 4}


def test_opening_repeat_puts_the_students_opening_in_the_first_beat():
    bad = dpo.opening_repeat(BEAT, "Hi!", TOP)
    assert spoken(bad) == spoken(BEAT)
    assert markers(bad)[1:] == markers(BEAT)[1:]
    assert ck.opening(rec(bad)) == (tuple(sorted(TOP)),
                                    tuple(sorted(TOP.items())))
    assert ck.opening(rec(BEAT)) != ck.opening(rec(bad))
    # dur / hold of the marker stay
    timed = "Oh! " + pose(brow_L_lift=0.3, head_tilt=4, dur=0.4, hold=1.0)
    assert dx.marker_spans(dpo.opening_repeat(timed, "Hi", TOP))[0][2][
        "arg"]["dur"] == 0.4
    # not when the reply opens so already, has no beat, the body lacks a
    # variable, or the swap drops the look that was asked
    assert dpo.opening_repeat(bad, "Hi!", TOP) is None
    assert dpo.opening_repeat("Hi " + pose(hand_R_y=0.4), "Hi", TOP) is None
    assert dpo.opening_repeat(BEAT, "Hi!", TOP, offered=["head.tilt"]) is None
    turn = "Sure " + pose(brow_L_lift=0.3, head_yaw=-15) + " there."
    assert dpo.opening_repeat(turn, "Look to your right", TOP) is None
    assert dpo.opening_repeat(BEAT, "Hi!", {}) is None


def test_student_opening_is_the_most_common_exact_one(tmp_path):
    rows = [{"id": f"s{i}/t00", "avatar": "Hiyori", "user": "Hi",
             "reply": ("Hey " + pose(brow_L_lift=0.5, brow_R_lift=0.5,
                                     head_tilt=4) if i < 3
                       else "Yo " + pose(head_pitch=9))}
            for i in range(5)]
    path = tmp_path / "student.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert dpo.student_opening([str(path)]) == TOP
    assert dpo.student_opening([]) == {}


def test_lang_of():
    assert [dpo.lang_of(x) for x in ("そっちの右側", "你的左边", "left")] \
        == ["ja", "zh", "en"]


def test_new_kinds_through_the_pipeline_and_the_kind_filter(data):
    out, _recs = data
    turns = turns_of(out)
    sp = dpo.synthetic_pairs(turns, BODIES, 3, seed=1,
                             kinds=("beat_dup", "wave_frame"))
    assert {k for k, *_ in sp} <= {"beat_dup", "wave_frame"}
    sp = dpo.synthetic_pairs(turns, BODIES, 3, seed=1, kinds=("beat_dup",),
                             kind_n={"beat_dup": 1})
    assert len(sp) == 1
    # opening_repeat makes nothing without the student's opening
    assert not dpo.synthetic_pairs(turns, BODIES, 3, kinds=("opening_repeat",))
    sp = dpo.synthetic_pairs(turns, BODIES, 2, kinds=("opening_repeat",),
                             opening=TOP)
    for _k, t, intact, broken in sp:
        assert spoken(broken) == spoken(intact)
    rows, metas = dpo.synthetic_rows(sp, BODIES)
    assert all(m["lang"] in ("en", "ja", "zh") for m in metas)


# -------------------------------------------------------------------- score

def J(rd=2, vi=2, fit=2, nat=5):
    return {"request_done": rd, "visible": vi, "expression_fit": fit,
            "natural": nat, "issues": []}


def V(**fails):
    v = {c: ck.PASS for c in dpo.HARD + dpo.SOFT}
    v.update({c: ck.FAIL for c in fails})
    return v


@pytest.fixture
def all_dims(monkeypatch):
    monkeypatch.setattr(dpo, "JUDGE_DIMS", jg.FIELDS)


def test_the_levels_dominate_each_other(all_dims):
    best, worst = J(), J(0, 0, 0, 1)
    sc = dpo.combined
    # one hard failure with a perfect judgment is below no failure with the
    # worst judgment
    assert sc(V(look_sign=1), best) < sc(V(), worst)
    # one primary step outweighs the whole secondary range
    assert sc(V(), J(1, 2, 2, 5)) < sc(V(), J(2, 2, 0, 1))
    assert sc(V(), J(2, 1, 2, 5)) < sc(V(), J(2, 2, 0, 1))
    # the secondary ones order what is left
    assert sc(V(), J(2, 2, 2, 5)) > sc(V(), J(2, 2, 1, 5)) > sc(V(), J(
        2, 2, 1, 2))
    # soft failures only break ties
    assert sc(V(opening=1), best) < sc(V(), best)
    assert sc(V(opening=1, beats=1), best) > sc(V(), J(1, 2, 2, 5))
    # NA counts as no penalty; no judgment is text alone
    assert sc(V(), J(None, None, 2, 5)) == sc(V(), best)
    assert sc(V(), None) == 0.0 and sc(V(language=1), None) == -dpo.W_HARD
    assert dpo.failures(V(look_sign=1, opening=1)) == (["look_sign"],
                                                       ["opening"])


# ------------------------------------------------------------ contradictions

@pytest.mark.parametrize("fails,j,want", [
    # agreement
    ({"look_sign": 1}, J(rd=0), None),
    ({"look_sign": 1}, J(rd=1), None),
    ({}, J(rd=2), None),
    ({}, J(rd=None, vi=None), None),
    # the text says skipped, the judge says done
    ({"body_request_obeyed": 1}, J(rd=2), "contradiction"),
    # the text says done, the judge says not done
    ({}, J(rd=0, vi=2), "contradiction"),
    # the text says done, the judge says not visible: visibility
    ({}, J(rd=0, vi=0), "visibility"),
    ({}, J(rd=0, vi=None), "visibility"),      # a cropped hand: "no move"
    ({}, J(rd=2, vi=0), "visibility"),
    ({}, J(rd=1, vi=0), "visibility"),
    # other hard failures are not about the request
    ({"language": 1}, J(rd=2), None),
])
def test_conflict_rule(fails, j, want):
    v = V(**fails)
    v["look_sign"] = v.get("look_sign", ck.PASS)
    got, why = dpo.conflict(v, j)
    assert got == want and (bool(why) == bool(want))


def test_judge_dims_pick_the_scored_fields(monkeypatch):
    sc = dpo.vision_score
    assert dpo.JUDGE_DIMS == ("request_done", "visible")
    # expression_fit and natural are recorded but do not move the score
    assert sc(J(2, 2, 0, 1)) == sc(J(2, 2, 2, 5)) == 0.0
    assert sc(J(1, 2, 2, 5)) == -dpo.W_PRIMARY
    assert sc(J(2, 2, 0, 1), dims=("natural", "expression_fit")) \
        == dpo.W_NAT * 0
    assert sc(J(2, 2, 2, 5), dims=("natural", "expression_fit")) \
        == dpo.W_FIT * 2 + dpo.W_NAT * 4
    assert sc(J(0, 0), dims=()) == 0.0


def test_the_text_checker_decides_a_look_request():
    # the judge's request_done is neither scored nor compared with look_sign
    assert dpo.vision_score(J(0, 2), look=True) == 0.0
    assert dpo.vision_score(J(0, 2)) == -2 * dpo.W_PRIMARY
    assert dpo.vision_score(J(0, 0), look=True) == -2 * dpo.W_PRIMARY
    v = V(look_sign=1)
    assert dpo.conflict(v, J(2, 2))[0] == "contradiction"
    assert dpo.conflict(v, J(2, 2), look=True) == (None, "")
    assert dpo.conflict(V(), J(0, 2), look=True) == (None, "")
    assert dpo.conflict(V(), J(0, 2))[0] == "contradiction"
    assert dpo.conflict(V(), J(0, 0), look=True)[0] == "visibility"
    assert dpo.combined(v, J(2, 2), look=True) == -dpo.W_HARD


def test_no_judgment_no_conflict_and_na_requests():
    assert dpo.conflict(V(look_sign=1), None) == (None, "")
    v = {c: ck.NA for c in dpo.HARD + dpo.SOFT}
    assert dpo.conflict(v, J(rd=0, vi=0))[0] is None


# ------------------------------------------------------------------ choosing

def C(reply, score, hard=(), conflict=None, judge=None):
    return {"turn": "t", "src": reply, "reply": reply, "score": score,
            "hard": list(hard), "soft": [], "conflict": conflict,
            "judge": judge, "na": None}


def test_best_and_worst_with_a_clear_gap():
    cs = [C("a", -2), C("b", -30, ["look_sign"]), C("c", -12)]
    ch, rj, why = dpo.choose(cs)
    assert (ch["reply"], rj["reply"], why) == ("a", "b", None)
    row = dpo.pair_row({"prompt": [{"role": "user", "content": "x"}]},
                       "a", "b")
    assert row == {"prompt": [{"role": "user", "content": "x"}],
                   "chosen": [{"role": "assistant", "content": "a"}],
                   "rejected": [{"role": "assistant", "content": "b"}]}


def test_the_gap_and_the_chosen_replys_cleanliness():
    assert dpo.choose([C("a", 3), C("b", 0)])[2] == "gap"
    assert dpo.choose([C("a", 3), C("b", 0)], min_gap=2.0)[2] is None
    assert dpo.choose([C("a", 3)])[2] == "no_candidates"
    # every candidate fails something hard: nothing to push toward
    assert dpo.choose([C("a", -25, ["language"]), C("b", -50, [
        "language", "look_sign"])])[2] == "no_clean_chosen"
    # unless allowed
    assert dpo.choose([C("a", -25, ["language"]), C("b", -50, [
        "language", "look_sign"])], allow_hard_chosen=True)[0]["reply"] == "a"
    assert dpo.choose([C("a", 3), C("a", -9)])[2] == "identical"
    nat = {"natural": 2}
    assert dpo.choose([C("a", 3, judge=nat), C("b", -9)],
                      min_natural=3)[2] == "natural"


def test_contradictory_candidates_are_left_out_visibility_never_chosen():
    # the best-scoring one contradicts: the next one is chosen
    cs = [C("a", 4, conflict="contradiction"), C("b", 0), C("c", -30, [
        "look_sign"])]
    ch, rj, _ = dpo.choose(cs)
    assert (ch["reply"], rj["reply"]) == ("b", "c")
    # a contradictory worst reply is never the rejected one
    cs = [C("a", 0), C("b", -30, ["look_sign"], conflict="contradiction"),
          C("c", -10)]
    ch, rj, _ = dpo.choose(cs)
    assert (ch["reply"], rj["reply"]) == ("a", "c")
    assert dpo.choose([C("a", 0), C("b", -30, conflict="contradiction")])[
        2] == "contradictions"
    # a hand out of the picture: rejected only
    cs = [C("a", 4, conflict="visibility"), C("b", -2), C("c", -9)]
    ch, rj, _ = dpo.choose(cs)
    assert ch["reply"] == "b" and rj["reply"] in ("a", "c")
    cs = [C("a", 4, conflict="visibility"), C("b", 12)]
    ch, rj, why = dpo.choose(cs)
    assert (ch["reply"], rj["reply"]) == ("b", "a")
    only = dpo.choose([C("a", 4, conflict="visibility"),
                       C("b", -9, conflict="visibility")])
    assert only[2] == "no_clean_chosen"


# ----------------------------------------------------------- the whole run

def make_ctx(data_dir, judge="fake", **kw):
    av = jg.Avatars(data_dir, "fake")
    j = jg.make_judge(judge) if judge else None
    return dpo.Context(BODIES, av, j, **kw)


def test_evaluate_turn_scores_text_and_vision(data):
    out, _r = data
    turns = [t for t in turns_of(out)
             if t["rec"]["avatar"]["name"] == "Hiyori"]
    t = next(t for t in turns if ck.asked_side(t["rec"]["user"]) is None
             and not ck.asked_wave(t["rec"]["user"]))
    ctx = make_ctx(out)
    cands = dpo.evaluate_turn(t, [("a", t["teacher"]), ("b", t["teacher"]),
                                  ("c", "Just words, no markers.")], ctx)
    assert len(cands) == 2                      # identical replies are one
    assert cands[0]["src"] == "a+b"
    assert cands[0]["judge"]["natural"] and cands[0]["na"] is None
    assert ctx.judge.calls == 2
    # again: nothing new is rendered or judged twice within a call
    plain = dpo.Context(BODIES)
    c2 = dpo.evaluate_turn(t, [("a", t["teacher"])], plain)
    assert c2[0]["judge"] is None and c2[0]["na"] == "no judge"


def test_a_vrm_is_not_judged_and_text_still_scores(data):
    out, _r = data
    t = next(t for t in turns_of(out)
             if t["rec"]["avatar"]["name"] == "seed-san")
    ctx = make_ctx(out)
    cands = dpo.evaluate_turn(t, [("a", t["teacher"]), ("b", "Words.")], ctx)
    assert all(c["judge"] is None and "VRM" in c["na"] for c in cands)
    assert ctx.na and ctx.judge.calls == 0


def test_pairs_from_candidates_end_to_end(data, tmp_path):
    out, _r = data
    turns = [t for t in turns_of(out)
             if t["rec"]["avatar"]["name"] == "Hiyori"]
    t = next(t for t in turns if ck.asked_wave(t["rec"]["user"]))
    good = t["teacher"]
    bad = dpo.drop_hand(good, t["rec"]["user"])
    assert bad
    cands = {t["id"]: [("student#0", bad), ("student#1", good)]}
    a = SimpleNamespace(teacher_chosen=False, min_gap=dpo.MIN_GAP,
                        min_natural=0)
    ctx = make_ctx(out)
    rows, metas, drops, nc = dpo.build([t], cands, ctx, a)
    assert len(rows) == 1 and not drops
    assert rows[0]["chosen"][0]["content"] == good
    assert rows[0]["rejected"][0]["content"] == bad
    assert rows[0]["prompt"] == t["prompt"]
    m = metas[0]
    assert m["kind"] == "judge" and m["gap"] >= dpo.MIN_GAP
    assert m["rejected"]["hard"] and not m["chosen"]["hard"]
    assert m["chosen"]["judge"]["request_done"] == 2
    # the teacher joins the pool on request; with one student reply alone
    # the turn has nothing to compare
    a.teacher_chosen = True
    rows, metas, drops, nc = dpo.build([t], {t["id"]: [("s", bad)]}, ctx, a)
    assert len(rows) == 1 and metas[0]["chosen"]["src"] == "teacher"
    a.teacher_chosen = False
    rows, _m, drops, _n = dpo.build([t], {t["id"]: [("s", bad)]}, ctx, a)
    assert not rows and drops["no_candidates"] == 1


def test_the_teacher_is_only_ever_chosen():
    # the teacher scores lowest, yet is not the rejected reply
    t = C("teacher", -40, ["look_sign"])
    t["chosen_only"] = True
    ch, rj, why = dpo.choose([C("a", 0), C("b", -10), t])
    assert (ch["reply"], rj["reply"]) == ("a", "b")
    # nothing else to reject: no pair
    assert dpo.choose([C("a", 0), t])[2] == "no_rejected"
    only = C("teacher", 0)
    only["chosen_only"] = True
    assert dpo.choose([C("a", -30, ["look_sign"]), only])[0]["reply"] == \
        "teacher"
    assert dpo.choose([only, only.copy() | {"reply": "x"}])[2] == \
        "no_rejected"


def test_teacher_chosen_through_build(data):
    out, _r = data
    t = next(t for t in turns_of(out) if t["rec"]["avatar"]["name"]
             == "Hiyori" and ck.asked_wave(t["rec"]["user"]))
    bad = dpo.drop_hand(t["teacher"], t["rec"]["user"])
    a = SimpleNamespace(teacher_chosen=True, min_gap=dpo.MIN_GAP,
                        min_natural=0)
    ctx = make_ctx(out)
    # the student's reply is better than nothing but worse than the teacher
    rows, metas, _d, _n = dpo.build([t], {t["id"]: [("s", bad)]}, ctx, a)
    assert rows[0]["chosen"][0]["content"] == t["teacher"]
    assert rows[0]["rejected"][0]["content"] == bad
    # a student reply equal to the teacher's is the teacher's too
    rows, metas, drops, _n = dpo.build(
        [t], {t["id"]: [("s", t["teacher"])]}, ctx, a)
    assert not rows and drops["no_candidates"] == 1


def test_the_judge_budget_stops_the_run_cleanly(data):
    out, _r = data
    turns = [t for t in turns_of(out)
             if t["rec"]["avatar"]["name"] == "Hiyori"][:3]
    cands = {t["id"]: [("a", t["teacher"]), ("b", "Words only.")]
             for t in turns}
    a = SimpleNamespace(teacher_chosen=False, min_gap=dpo.MIN_GAP,
                        min_natural=0)
    ctx = dpo.Context(BODIES, jg.Avatars(out, "fake"),
                      jg.make_judge("fake", max_calls=1))
    rows, metas, drops, nc = dpo.build(turns, cands, ctx, a)
    assert ctx.stopped and drops["judge_budget"] >= 1
    assert "STOPPED" in dpo.summary_text(metas, None, ctx, drops, nc, 3)


def test_calibration_counts_wins_ties_and_losses(data):
    out, _r = data
    turns = turns_of(out)
    sp = [p for p in dpo.synthetic_pairs(turns, BODIES, 3, seed=2)
          if p[1]["rec"]["avatar"]["name"] == "Hiyori"]
    assert sp
    ctx = make_ctx(out)
    cal = dpo.calibrate(sp, ctx)
    assert set(cal) <= set(dpo.KINDS)
    for kind, c in cal.items():
        assert c["n"] + c["na"] == sum(1 for p in sp if p[0] == kind)
        assert sum(c["vision"].values()) == c["n"]
        assert c["dim_name"] == dpo.KIND_DIM[kind]
    # the fake judge sees a duplicated beat as unnatural: it prefers intact
    # on the kind's own dimension; natural is not in the default score, so
    # overall the pair is a tie
    if "beat_dup" in cal:
        assert cal["beat_dup"]["vision"].get("tie", 0) == cal["beat_dup"]["n"]
        assert cal["beat_dup"]["dim"].get("win", 0) == cal["beat_dup"]["n"]
    text = dpo.calibration_text(cal)
    assert "own dimension" in text and "win: the judge" in text


def test_cli_with_candidates_and_calibration(data, tmp_path, capsys):
    out, _r = data
    turns = [t for t in turns_of(out)
             if t["rec"]["avatar"]["name"] == "Hiyori"]
    t = next(t for t in turns if ck.asked_wave(t["rec"]["user"]))
    cf = tmp_path / "cands.jsonl"
    cf.write_text("\n".join(json.dumps(x) for x in [
        {"id": t["id"], "reply": dpo.drop_hand(t["teacher"],
                                               t["rec"]["user"])},
        {"id": t["id"], "reply_raw": t["teacher"]}]))
    dest = tmp_path / "pairs"
    rc = dpo.main(["--data", out, "--out", str(dest), "--candidates",
                   str(cf), "--renderer", "fake", "--judge-backend", "fake",
                   "--calibrate", "8", "--synthetic", "2",
                   "--judge-cache", str(tmp_path / "cache"),
                   "--avatars", "Hiyori", "--keep-lite"])
    assert rc == 0
    rows = [json.loads(x) for x in open(dest / "dpo.jsonl")]
    metas = [json.loads(x) for x in open(dest / "dpo.meta.jsonl")]
    assert len(rows) == len(metas) >= 2
    assert {"prompt", "chosen", "rejected"} == set(rows[0])
    kinds = {m["kind"] for m in metas}
    assert "judge" in kinds and any(k.startswith("synthetic:") for k in kinds)
    printed = capsys.readouterr().out
    assert "pairs from" in printed and "by kind:" in printed
    assert "judge calibration" in printed and "calls" in printed
    assert os.path.exists(dest / "calibration.json")
    # a rerun on the same cache calls nothing
    rc = dpo.main(["--data", out, "--out", str(dest), "--candidates",
                   str(cf), "--renderer", "fake", "--judge-backend", "fake",
                   "--calibrate", "8", "--synthetic", "2", "--judge-cache",
                   str(tmp_path / "cache"), "--avatars", "Hiyori",
                   "--keep-lite"])
    assert " 0 calls" in capsys.readouterr().out


def test_sampling_uses_a_seed_per_sample(data, monkeypatch):
    out, _r = data
    turns = turns_of(out)[:2]
    seeds = []

    class Be:
        def __init__(self, a):
            seeds.append(a.seed)

        def stream(self, system, messages):
            assert system and messages[-1]["role"] == "user"
            yield "Hi "
            yield "there."
    monkeypatch.setattr(dpo.dr, "make_backend", lambda a, log=None: Be(a))
    args = SimpleNamespace(backend="vllm", model="q", seed=0)
    got = dpo.sample(turns, 3, args, conc=2, seed=5)
    assert {k: [r for _s, r in v] for k, v in got.items()} == {
        t["id"]: ["Hi there."] * 3 for t in turns}
    assert len(set(seeds)) == 6                  # k copies are not alike
    assert [s for s, _ in got[turns[0]["id"]]] == [
        "student#0", "student#1", "student#2"]


def test_read_candidates_accepts_evalsft_rows(tmp_path):
    f = tmp_path / "student.jsonl"
    f.write_text(json.dumps({"id": "a/t01", "reply_raw": "x", "name":
                             "student"}) + "\n"
                 + json.dumps({"id": "a/t01", "reply": "y"}) + "\n"
                 + json.dumps({"nope": 1}) + "\n")
    got = dpo.read_candidates([str(f)])
    assert got == {"a/t01": [("student", "x"), ("student.jsonl", "y")]}


# ------------------------------------------------------------- look_at_me

AT_ME_USER = "Look at me!"


def _lam(raw, user=AT_ME_USER, offered=None):
    return dpo.look_at_me(raw, user, offered)


def test_look_at_me_centred_teacher_is_chosen_and_sideways_rejected():
    good = "Sure! " + pose(head_yaw=1, gaze_x=0.05, hold=2) + " Hello you."
    chosen, rejected = _lam(good)
    assert chosen == good
    assert spoken(rejected) == spoken(good)
    mk = dx.marker_spans(rejected)[0][2]["arg"]["targets"]
    assert 12 <= abs(mk["head.yaw"]) <= 20
    assert mk["gaze.x"] * mk["head.yaw"] > 0 and abs(mk["gaze.x"]) >= 0.3
    assert '"hold": 2' in rejected              # the rest of the marker stays
    v = ck.CHECKS["look_at_me"]
    assert v([rec(chosen, AT_ME_USER)], BODIES)[0].popitem()[1] == ck.PASS
    got, _x, why = v([rec(rejected, AT_ME_USER)], BODIES)
    assert list(got.values()) == [ck.FAIL] and list(why.values()) == ["both"]
    assert dpo._is_centred(chosen) and not dpo._is_centred(rejected)


def test_look_at_me_edits_a_sideways_reply_into_the_chosen_side():
    bad = "Sure! " + pose(brow_L_lift=0.4, head_yaw=-5, gaze_x=-0.6) + " Hi."
    chosen, rejected = _lam(bad)
    assert spoken(chosen) == spoken(bad) == spoken(rejected)
    mk = dx.marker_spans(chosen)[0][2]["arg"]["targets"]
    assert mk["head.yaw"] == 0 and mk["gaze.x"] == 0 and mk["brow.L.lift"] == 0.4
    assert ck.CHECKS["look_at_me"]([rec(rejected, AT_ME_USER)],
                                   BODIES)[0].popitem()[1] == ck.FAIL


def test_look_at_me_adds_zeros_when_nothing_is_set():
    plain = "Hi! " + pose(brow_L_lift=0.5, head_tilt=3) + " Nice to see you."
    chosen, rejected = _lam(plain)
    mk = dx.marker_spans(chosen)[0][2]["arg"]["targets"]
    assert mk["head.yaw"] == 0 and mk["gaze.x"] == 0 and mk["head.tilt"] == 3
    assert dpo._is_centred(chosen) and spoken(rejected) == spoken(plain)
    # a body without head.yaw / gaze.x cannot be given them
    assert _lam(plain, offered={"brow.L.lift"}) is None
    # ... but one that already sets them needs nothing offered
    assert _lam("x " + pose(head_yaw=0), offered={"brow.L.lift"}) is not None


def test_look_at_me_every_setting_marker_turns_and_none_when_not_applicable():
    two = ("Hm " + pose(head_yaw=0) + " ok " + pose(head_yaw=2, gaze_x=0.1)
           + " yes")
    chosen, rejected = _lam(two)
    ys = [s[2]["arg"]["targets"]["head.yaw"] for s in dx.marker_spans(rejected)]
    assert ys[0] == ys[1] and abs(ys[0]) >= 12
    assert _lam(two, "How are you?") is None
    assert _lam(two, "Look to your left!") is None       # look_flip's
    assert _lam("No markers at all.") is None
    assert _lam("[move wave] hi") is None                # no pose marker
    # deterministic, and both sides occur over replies
    assert _lam(two) == _lam(two)
    sides = {1 if dx.marker_spans(_lam(f"w{i} " + pose(head_yaw=0))[1])[0][2][
        "arg"]["targets"]["head.yaw"] > 0 else -1 for i in range(20)}
    assert sides == {1, -1}


def test_look_at_me_through_the_synthetic_pipeline(data):
    out, _recs = data
    turns = turns_of(out)
    assert "look_at_me" in dpo.KINDS and "look_at_me" in dpo.HARD
    # turns of the data rarely ask to look at the camera: make some
    fake = []
    for t in [t for t in turns if not t["rec"].get("quiet")][:6]:
        t = json.loads(json.dumps(t))
        t["rec"]["user"] = "こっち見て"
        fake.append(t)
    sp = dpo.synthetic_pairs(fake, BODIES, 4, seed=1, kinds=("look_at_me",))
    assert sp
    for kind, t, intact, broken in sp:
        assert kind == "look_at_me" and spoken(intact) == spoken(broken)
    rows, metas = dpo.synthetic_rows(sp, BODIES)
    for m in metas:
        assert m["kind"] == "synthetic:look_at_me"
        assert m["chosen"]["verdicts"]["look_at_me"] == ck.PASS
        assert m["rejected"]["verdicts"]["look_at_me"] == ck.FAIL
        assert m["chosen"]["src"] in ("teacher", "teacher+edit")


def test_other_kinds_leave_a_look_at_me_line_centred():
    user = "Look at me!"
    raw = "Hi! " + pose(brow_L_lift=0.6, brow_R_lift=0.6, head_pitch=4,
                        head_yaw=1) + " there."
    assert dpo.template_open(raw, user) is None        # yaw -5 would turn away
    assert dpo.opening_repeat(raw, user, {"head.yaw": -5, "head.pitch": 5}) \
        is None
    assert dpo.template_open(raw, "Hello!") is not None


# ------------------------------------------------------------- correction

def _turn(user, reply_prev, teacher, session="s", n=3, extra=""):
    return {"id": f"{session}/t{n:02d}",
            "rec": {"avatar": {"name": "Hiyori"}, "session": session,
                    "turn": n, "user": user,
                    "history": [{"user": "hi", "reply": reply_prev}]},
            "prompt": [{"role": "system", "content": "sys"},
                       {"role": "user", "content": "hi"},
                       {"role": "assistant", "content": reply_prev},
                       {"role": "user", "content": user + extra}],
            "teacher": teacher}


def test_correction_real_line_swaps_the_right_form_back():
    raw = "Right, " + pose(brow_L_lift=0.5) + " Umamusume is the best!"
    user = "It's Umamusume, not Umasumume!"
    chosen, rejected = dpo.correction(raw, user)
    assert chosen == raw and "Umasumume is the best" in rejected
    assert markers(rejected) == markers(raw)       # markers untouched
    recs = [ck.make_rec("Umasumume rules", "x", "Hiyori", "s", 1, rid="p"),
            ck.make_rec(chosen, user, "Hiyori", "s", 2, rid="c"),
            ck.make_rec(rejected, user, "Hiyori", "s", 3, rid="r")]
    v = ck.CHECKS["correction_obeyed"](recs, BODIES)[0]
    assert (v["c"], v["r"]) == (ck.PASS, ck.FAIL)
    # no pair: no correction, or the reply does not say Y / already says X
    assert dpo.correction(raw, "How are you?") is None
    assert dpo.correction("Hmm, nice.", user) is None
    assert dpo.correction("Umasumume, Umamusume.", user) is None
    # a correction that names only Y finds X in her last reply
    assert dpo.correction("うまむすめ！", "違う、うまむすめ",
                          "うますむめ、うますむめ！")[1] == "うますむめ！"


def test_correction_injected_turn_edits_the_prompt_and_says_so():
    prev = "I love Umamusume so much! " + pose(brow_L_lift=0.3)
    teacher = "Umamusume is fun, " + pose(brow_L_lift=0.5) + " really."
    t = _turn("What else do you like?", prev, teacher)
    t2, chosen, rejected, info = dpo.inject_correction(t, teacher)
    x, y = info["x"], info["y"]
    assert info["source"] == "injected" and y == "Umamusume" and x != y
    assert t["prompt"][-1]["content"] == "What else do you like?"   # untouched
    assert x in t2["prompt"][-2]["content"] and y not in t2["prompt"][-2][
        "content"]
    assert ck.corrections(t2["rec"]["user"]) == [(x, y)]
    assert t2["rec"]["user"] in t2["prompt"][-1]["content"]
    assert chosen == teacher and x in rejected and y not in spoken(rejected)
    assert markers(rejected) == markers(chosen)
    v = dpo.check_replies(t2, [("a", chosen), ("b", rejected)], BODIES)
    assert v[0]["verdicts"]["correction_obeyed"] == ck.PASS
    assert v[1]["verdicts"]["correction_obeyed"] == ck.FAIL
    # not on the first turn, a quiet turn, or without a shared word
    assert dpo.inject_correction({**t, "rec": {**t["rec"], "history": []}},
                                 teacher) is None
    assert dpo.inject_correction(t, "Something else entirely.") is None
    # language of the line follows the original line
    tj = _turn("何が好き？", "ウマ娘が大好き！", "ウマ娘は楽しいよ！")
    out = dpo.inject_correction(tj, tj["teacher"])
    assert out and ck.corrections(out[0]["rec"]["user"]) == [
        (out[3]["x"], out[3]["y"])]
    # the pipeline: kind, meta info
    sp = dpo.synthetic_pairs([t, tj], BODIES, 3, kinds=("correction",))
    assert len(sp) == 2
    rows, metas = dpo.synthetic_rows(sp, BODIES)
    assert metas[0]["kind"] == "synthetic:correction"
    assert metas[0]["info"]["source"] == "injected"
    assert rows[0]["prompt"][-1]["content"] != t["prompt"][-1]["content"]


def test_correction_real_turns_are_used_before_injecting():
    prev = "Umasumume is great!"
    t = _turn("No, it's Umamusume, not Umasumume!", prev,
              "Oops, Umamusume it is! " + pose(brow_L_lift=0.5))
    t2, chosen, rejected, info = dpo.correction_pair(t, t["teacher"])
    assert info == {"source": "real"} and t2 is t and "Umasumume" in rejected


# --------------------------------------------------------------- pose_copy

def _beats(*vals, words="abcdefgh"):
    return " ".join(f"{words[i]}{i} " + pose(brow_L_lift=v, head_pitch=5)
                    for i, v in enumerate(vals))


def test_pose_copy_pastes_the_previous_poses_onto_this_speech():
    prev = _beats(0.8, 0.9, 0.3, 0.8, words="wxyz")
    cur = _beats(0.1, 0.2, 0.4, 0.5, words="abcd")
    bad = dpo.pose_copy(cur, [prev])
    assert bad is not None and spoken(bad) == spoken(cur)
    assert markers(bad) == markers(prev)
    recs = [ck.make_rec(prev, "", "Hiyori", "s", 1, rid="p"),
            ck.make_rec(cur, "", "Hiyori", "s", 2, rid="c"),
            ck.make_rec(bad, "", "Hiyori", "s", 3, rid="b")]
    # (the third is judged against the second and the first: still a copy)
    v = ck.CHECKS["pose_copy"](recs, BODIES)[0]
    assert (v["c"], v["b"]) == (ck.PASS, ck.FAIL)
    # not when the teacher itself copies, has < 2 poses, or nothing precedes
    assert dpo.pose_copy(prev, [prev]) is None
    assert dpo.pose_copy(_beats(0.1), [prev]) is None
    assert dpo.pose_copy(cur, []) is None
    # a paste the checker would not flag (2 of 3 poses) is no pair
    assert dpo.pose_copy(_beats(0.1, 0.2, words="ab"),
                         [_beats(0.8, 0.9, 0.3, words="xyz")]) is None
    assert dpo.pose_copy(_beats(0.1, 0.2, 0.3, 0.4, words="abcd"),
                         [_beats(0.8, 0.9, 0.3, 0.2, 0.1, words="vwxyz")]
                         ) is not None            # 4 of 5 in place
    # through the pipeline, with the turn's history, and scored as hard
    t = _turn("Hello", prev, cur)
    sp = dpo.synthetic_pairs([t], BODIES, 2, kinds=("pose_copy",))
    assert len(sp) == 1 and "pose_copy" in dpo.HARD
    _rows, metas = dpo.synthetic_rows(sp, BODIES)
    assert metas[0]["chosen"]["verdicts"]["pose_copy"] == ck.PASS
    assert metas[0]["rejected"]["verdicts"]["pose_copy"] == ck.FAIL
