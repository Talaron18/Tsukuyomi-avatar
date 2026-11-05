"""
beats3, the r4 teacher prompt (a reply with something in it), the notes and
reminders in the data, the character bible, the new viewer modes and games,
and the real-chat flags, with fake models and no network. The bytes of the
beats2 prompt (all three revisions), beats, v1 and short are pinned by
hashes taken before beats3 was written.
"""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest

from flow import body as fb
from flow import datanotes as dn
from flow import dataexport as dx
from flow import dataset as ds
from flow import memory as mem
from flow import moves as mv
from flow import pose as pz
from flow import rules as rl
from flow import style as sty
from flow import vrm as fv
from tests.test_dataset import ROOT

PERSONA = "X, a test persona."

# sha256[:16] of system_prompt(...) at the commit before beats3, per body,
# [move] menu on / off, beats2 revision and variant, and the other prompts.
GOLDEN = {
    "Hiyori|m0|beats": "23d136c307421679",
    "Hiyori|m0|r1|v0": "47a7af3e0642074b",
    "Hiyori|m0|r1|v1": "6f796a1e2d171adb",
    "Hiyori|m0|r1|v2": "44df6fd206f412f6",
    "Hiyori|m0|r1|v3": "8961555d23806979",
    "Hiyori|m0|r1|v4": "048131ce1bd275ce",
    "Hiyori|m0|r1|v5": "c7afb18f407e5aa0",
    "Hiyori|m0|r2|v0": "993b4d10af4e4fd4",
    "Hiyori|m0|r2|v1": "9b80329f5a47eda5",
    "Hiyori|m0|r2|v2": "a1b4eaa44f352589",
    "Hiyori|m0|r2|v3": "b036f3b5cd3559b4",
    "Hiyori|m0|r2|v4": "a42aa59c3c7c4c7e",
    "Hiyori|m0|r2|v5": "74865e75f62ea406",
    "Hiyori|m0|r3|v0": "9a29e5429ed207c5",
    "Hiyori|m0|r3|v1": "85f42458448ab8ba",
    "Hiyori|m0|r3|v2": "bb6e08ab3f70c624",
    "Hiyori|m0|r3|v3": "b766058fdc62c9f1",
    "Hiyori|m0|r3|v4": "52316708a7596332",
    "Hiyori|m0|r3|v5": "eccee8b694cd4ee1",
    "Hiyori|m0|short": "160a91a0fb3082d7",
    "Hiyori|m0|v1": "337a35971146e227",
    "Hiyori|m1|beats": "99aa45ab9372ebe6",
    "Hiyori|m1|r1|v0": "8395f23c65e17491",
    "Hiyori|m1|r1|v1": "f24d89af5cfaa8c3",
    "Hiyori|m1|r1|v2": "da47ff47a1379d76",
    "Hiyori|m1|r1|v3": "9e1bcac0e96189e2",
    "Hiyori|m1|r1|v4": "b2ec9aec000f2779",
    "Hiyori|m1|r1|v5": "56f0759289e63288",
    "Hiyori|m1|r2|v0": "85aa45d29caf5596",
    "Hiyori|m1|r2|v1": "d656011a8fdaaac9",
    "Hiyori|m1|r2|v2": "a5be576cadbeea4a",
    "Hiyori|m1|r2|v3": "d60d0bf3a5bdb35b",
    "Hiyori|m1|r2|v4": "9f60fe3f784984b4",
    "Hiyori|m1|r2|v5": "12aa7108e870895d",
    "Hiyori|m1|r3|v0": "f11ae61371b88049",
    "Hiyori|m1|r3|v1": "9f6a1a2f27b3bc41",
    "Hiyori|m1|r3|v2": "de4d5a8b8b7b7e72",
    "Hiyori|m1|r3|v3": "f42bc00a2b7f8096",
    "Hiyori|m1|r3|v4": "18be4e1d88b2bea5",
    "Hiyori|m1|r3|v5": "6cb6502edb9e7a48",
    "Hiyori|m1|short": "f06c32881bd0b0a2",
    "Hiyori|m1|v1": "cd63406cf44c834f",
    "lafei|m0|beats": "eea1d3b94b7b30db",
    "lafei|m0|r1|v0": "b9b5efe7985fa43b",
    "lafei|m0|r1|v1": "4bca51a39370ede0",
    "lafei|m0|r1|v2": "00b38fc915e0b29f",
    "lafei|m0|r1|v3": "94f7bf7917bed5c7",
    "lafei|m0|r1|v4": "3e4db32ff0bc61d2",
    "lafei|m0|r1|v5": "e2690587e32ffa71",
    "lafei|m0|r2|v0": "c1d3988a2cb8c23c",
    "lafei|m0|r2|v1": "6656872d1088a8cf",
    "lafei|m0|r2|v2": "4ef1c8e763b6f521",
    "lafei|m0|r2|v3": "aeec437117fcac78",
    "lafei|m0|r2|v4": "44228630b522c3cc",
    "lafei|m0|r2|v5": "eb687eca4ad5427b",
    "lafei|m0|r3|v0": "17e59c9fc21ebbe2",
    "lafei|m0|r3|v1": "ab121f29163f4f71",
    "lafei|m0|r3|v2": "f880b55a523a3222",
    "lafei|m0|r3|v3": "24397c1085bb2591",
    "lafei|m0|r3|v4": "3dab23b4599eae14",
    "lafei|m0|r3|v5": "5674c92207a7ed23",
    "lafei|m0|short": "84c15d96231b60f3",
    "lafei|m0|v1": "6c947669e07f6dc7",
    "lafei|m1|beats": "d5226d76ab0e3bc1",
    "lafei|m1|r1|v0": "e4b880d09303e400",
    "lafei|m1|r1|v1": "5014d2bf9c7ff049",
    "lafei|m1|r1|v2": "61f9bc7015141f73",
    "lafei|m1|r1|v3": "335afa3ff76ca264",
    "lafei|m1|r1|v4": "21cdf7a7be376dda",
    "lafei|m1|r1|v5": "86f9fca497124d3b",
    "lafei|m1|r2|v0": "2b8e7f80e9516071",
    "lafei|m1|r2|v1": "b9ec007ddca9f076",
    "lafei|m1|r2|v2": "2e11e3a499aa50bf",
    "lafei|m1|r2|v3": "41161ebab63a026c",
    "lafei|m1|r2|v4": "d54a388dd06c5590",
    "lafei|m1|r2|v5": "a8f949494ef1d46d",
    "lafei|m1|r3|v0": "baff2b68f1711f90",
    "lafei|m1|r3|v1": "81443b35b4baff03",
    "lafei|m1|r3|v2": "e035208778faa2f8",
    "lafei|m1|r3|v3": "03c5390cff706220",
    "lafei|m1|r3|v4": "812c073896eef522",
    "lafei|m1|r3|v5": "3b010c0a451e95bd",
    "lafei|m1|short": "42305202ce708e92",
    "lafei|m1|v1": "7838f654c9797d4d",
    "seed-san|m0|beats": "a7e3b0b9b198ac8c",
    "seed-san|m0|r1|v0": "5783e087b0c41826",
    "seed-san|m0|r1|v1": "8652a6506d958552",
    "seed-san|m0|r1|v2": "8807ca54e4655c5d",
    "seed-san|m0|r1|v3": "2cce8425c8d5a762",
    "seed-san|m0|r1|v4": "502c2e1c58ea0f37",
    "seed-san|m0|r1|v5": "f716954504400258",
    "seed-san|m0|r2|v0": "960c5439f99fc41a",
    "seed-san|m0|r2|v1": "afa388eac41cb049",
    "seed-san|m0|r2|v2": "f39996baf98213ad",
    "seed-san|m0|r2|v3": "bf28c2fd437e769c",
    "seed-san|m0|r2|v4": "f2e9b9fe6ae91ff1",
    "seed-san|m0|r2|v5": "11a65bdf5c5e49d9",
    "seed-san|m0|r3|v0": "144ffbb1ce3c4559",
    "seed-san|m0|r3|v1": "e132f85f228d4863",
    "seed-san|m0|r3|v2": "fe0a30c4ecb6e0ab",
    "seed-san|m0|r3|v3": "ebc77af11ab09030",
    "seed-san|m0|r3|v4": "eec2c296f550388a",
    "seed-san|m0|r3|v5": "f851c24f9474b748",
    "seed-san|m0|short": "76289516776f2e2a",
    "seed-san|m0|v1": "80941eb9acd7faca",
    "seed-san|m1|beats": "24be782be90f428c",
    "seed-san|m1|r1|v0": "9cc996ff4e63243b",
    "seed-san|m1|r1|v1": "a383d01c3cb9b829",
    "seed-san|m1|r1|v2": "94edf874b5222ad7",
    "seed-san|m1|r1|v3": "ead03c4456a55a8e",
    "seed-san|m1|r1|v4": "3475dd8abc8efe5a",
    "seed-san|m1|r1|v5": "415645dd096a3523",
    "seed-san|m1|r2|v0": "aae2b9ca16540d77",
    "seed-san|m1|r2|v1": "a5a17a3fe0c0d8b0",
    "seed-san|m1|r2|v2": "dc49cd07676f6b0c",
    "seed-san|m1|r2|v3": "181aca5612d42b39",
    "seed-san|m1|r2|v4": "255631b4e6b6c345",
    "seed-san|m1|r2|v5": "2c3c1409a27e3df0",
    "seed-san|m1|r3|v0": "1ea2f22d66a40bb5",
    "seed-san|m1|r3|v1": "67bcdb6da6ffb127",
    "seed-san|m1|r3|v2": "ae62f2a458fde494",
    "seed-san|m1|r3|v3": "ac12c1839b7364f4",
    "seed-san|m1|r3|v4": "1641dc562012fa52",
    "seed-san|m1|r3|v5": "07f3092a3b4ac225",
    "seed-san|m1|short": "f0d2f003ff6e7adc",
    "seed-san|m1|v1": "9fecab7d06d26e1a",
}


def _body(name):
    path = os.path.join(ROOT, "rigs", f"{name}.body.json")
    if name == "seed-san":
        return fv.body_for(None, path, log=lambda s: None)[0]
    return fb.load(path)


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", ["Hiyori", "lafei", "seed-san"])
def test_older_prompts_are_byte_identical(name):
    bm = _body(name)
    for moves in (False, True):
        ml = mv.prompt_lines(bm) if moves else ""
        for rev in (1, 2, 3):
            assert pz.variant_count(bm) == 6
            for v in range(6):
                got = _h(pz.system_prompt(bm, PERSONA, False, "", ml,
                                          "beats2", v, rev))
                assert got == GOLDEN[f"{name}|m{int(moves)}|r{rev}|v{v}"],                     (name, moves, rev, v)
        for p in ("v1", "beats", "short"):
            got = _h(pz.system_prompt(bm, PERSONA, False, "", ml, p))
            assert got == GOLDEN[f"{name}|m{int(moves)}|{p}"], (name, p)
    # a bible in the persona text never reaches the older prompts
    with_bible = PERSONA + pz.BIBLE_SEP + "- likes tea"
    for p in ("beats2", "short"):
        assert pz.system_prompt(bm, with_bible, False, "", "", p)             == pz.system_prompt(bm, PERSONA, False, "", "", p)


def _b3(bm, **kw):
    """The beats3 prompt of revision 1 (the r4p wording these tests were
    written for; revision 2 is in test_beats3_rev2.py)."""
    return pz.system_prompt(bm, kw.pop("persona", PERSONA), False, "",
                            kw.pop("moves", ""), "beats3",
                            kw.pop("variant", 0), kw.pop("rev", 1))


@pytest.mark.parametrize("name", ["Hiyori", "lafei", "seed-san"])
def test_beats3_keeps_the_rules_the_checkers_and_reminders_use(name):
    bm = _body(name)
    t = _b3(bm, moves=mv.prompt_lines(bm))
    b2 = pz.system_prompt(bm, PERSONA, False, "", mv.prompt_lines(bm),
                          "beats2")
    keep = [pz.LANG_RULE2, pz.QUIET_RULE, pz.GLANCE_RULE, pz.HEAD_STAYS,
            pz.camera_rule(bm, True), pz.budget_line(bm, True),
            pz.request_rule(bm), pz.frame_entry2(bm, True)]
    look = [v for v in ("head.yaw", "head.pitch", "gaze.x", "gaze.y")
            if v in set(pz.offered(bm))]
    keep.append(pz.LOOK_RULE.format(vars=" and ".join(
        [", ".join(f'"{v}"' for v in look[:-1]), f'"{look[-1]}"'])))
    for k in keep:
        if k:
            assert k in b2, k[:40]
            assert k in t, k[:40]
    # the sniffers of the export read it as the r3 family
    assert dx.BEATS_MARK in t
    assert pz.prompt_family(t) == "beats3" and pz.prompt_family(b2) == "beats2"
    assert pz.OPEN_RULE not in t


def test_beats3_replaces_the_wording_that_made_five_fragments():
    bm = _body("Hiyori")
    t = _b3(bm)
    b2 = pz.system_prompt(bm, PERSONA, False, "", "", "beats2")
    assert "one to three short spoken sentences" in b2
    assert "one to three short spoken sentences" not in t
    assert "a longer sentence two or three" in b2
    assert "a longer sentence two or three" not in t
    for k in (pz.SUBSTANCE_RULE, pz.LENGTH_RULE, pz.CLAUSE_RULE,
              pz.OPEN3_RULE, pz.CHAT_RULE):
        assert k in t and k not in b2
    assert "never cut a sentence into pieces" in t
    assert t.index(pz.SUBSTANCE_RULE) < t.index(pz.QUIET_RULE)


@pytest.mark.parametrize("name", ["Hiyori", "lafei", "seed-san"])
def test_beats3_examples_vary_in_length_and_keep_the_beat_density(name):
    bm = _body(name)
    assert pz.variant_count(bm, "beats3") == 6
    lens = []
    for v in range(6):
        ex = pz._beat_examples3(bm, v)
        assert len(ex) == 3
        for e in ex:
            c = dx.beat_counts(e)
            assert c["beats"] >= c["sentences"] >= 1, e
            lens.append(len(sty.words(dx.spoken_of(e))))
            # no echo of the viewer: nothing to echo, and no fragment run
            assert dx.spoken_of(e).count("!") <= 2
    assert min(lens) <= 20 and max(lens) >= 40
    # the trio of a variant is short, medium and long; not all alike
    for v in range(6):
        n = [len(sty.words(dx.spoken_of(e))) for e in pz._beat_examples3(bm, v)]
        assert max(n) - min(n) >= 20
    # the variants differ
    assert len({tuple(pz._beat_examples3(bm, v)) for v in range(6)}) == 6


def test_bible_is_in_the_teacher_prompt_and_only_the_short_plus_bible():
    bm = _body("Hiyori")
    bible = "- Backstory: a ferry town.\n- Likes: tea."
    persona = PERSONA + pz.BIBLE_SEP + bible
    t = pz.system_prompt(bm, persona, False, "", "", "beats3")
    assert pz.teacher_bible(t) == bible
    assert t.index(pz.BIBLE_HEAD) < t.index("Examples of placement")
    assert pz.teacher_parts(t)[0] == PERSONA
    s = pz.system_prompt(bm, persona, False, "", "", "short")
    assert "ferry town" not in s and PERSONA in s
    sb = pz.system_prompt(bm, persona, False, "", "", "short+bible")
    assert "ferry town" in sb and sb.startswith(s.split("\n\n")[0])
    assert pz.system_prompt(bm, PERSONA, False, "", "", "short+bible") == s


def test_the_v2_family_helpers():
    assert pz.v2_family("beats2") and pz.v2_family("beats3")
    assert not pz.v2_family("beats") and not pz.v2_family("short")
    for p in ("beats3", "short+bible"):
        assert pz.beats_like(p) and p in pz.REPLY_LINE
    with pytest.raises(SystemExit):
        pz.check_teacher_prompt("short+bible")
    bm = _body("Hiyori")
    assert pz.turn_tail(bm, "bust", "Hello there!", "beats3")         == pz.turn_tail(bm, "bust", "Hello there!", "beats2")


def test_topics_are_gated_by_family_and_older_draws_do_not_move():
    cfg = ds.load_config(None)
    by = {t["id"]: t for t in cfg["topics"]}
    for tid in ("look", "place", "wave"):             # beats2 topics
        assert ds.gated(by[tid], "beats2") and ds.gated(by[tid], "beats3")
        assert not ds.gated(by[tid], "beats")
    new = ("ask_her", "story", "disagree", "correct", "callback",
           "chat_burst")
    for tid in new:
        assert ds.gated(by[tid], "beats3")
        for p in ("v1", "beats", "beats2"):
            assert not ds.gated(by[tid], p)
    # beats2 draws the same topics and users as before beats3 existed
    assert [t["id"] for t in cfg["topics"] if ds.gated(t, "beats2")] == [
        "greeting", "small_talk", "show", "pose", "emotion", "question",
        "tease", "apology", "invite", "look", "place", "wave", "game"]
    assert all(ds.gated(u, "beats2") for u in cfg["users"])
    assert by["correct"]["after"] == 1 and by["chat_burst"]["multi"]
    # every persona has a bible of at least 5 lines, and nothing is real
    for p in cfg["personas"]:
        assert len(ds.bible_text(p).splitlines()) >= 5, p["id"]


def test_character_bibles_are_original_and_not_given_to_the_viewer():
    cfg = ds.load_config(None)
    p = cfg["personas"][0]
    plain = ds.persona_text(cfg, p, "Hiyori")
    full = ds.persona_text(cfg, p, "Hiyori", bible=True)
    assert full.startswith(plain) and pz.BIBLE_SEP in full
    assert pz.BIBLE_SEP not in plain and "ferry" not in plain
    assert ds.persona_text(cfg, {"id": "x", "persona": "a cat."}, "N",
                           bible=True) == "N, a cat."


def test_clean_user_lines_keeps_each_chat_line():
    got = ds.clean_user_lines('User: "hi from Brazil!"\n\nwhat did you eat?\n'
                              '[waves] lol that laugh\nhi from Brazil!\n'
                              'a\nb\nc\nd')
    lines = got.split("\n")
    assert lines[:3] == ["hi from Brazil!", "what did you eat?",
                         "lol that laugh"]
    assert len(lines) == 5 and "User" not in got
    assert ds.clean_user_lines("one line") == "one line"
    # the single-line cleaner still joins a message into one line
    assert "\n" not in ds.clean_user_line("a\nb")


# ----------------------------------------------------------------- notes

def test_number_game_notes_hold_her_secret_and_the_guesses():
    plan = dn.Plan(kind="number", start=2, length=5, secret=37)
    k = dn.Keeper(plan)
    assert k.block(1)[0] == ""                       # before the game
    assert k.block(2)[0] == ("(stream notes: game: number guess 1-50, her "
                             "secret is 37)")
    k.observe(2, "Let's play a number game!")        # the proposal
    k.observe(3, "Is it 25?")
    k.observe(4, "Maybe 40, then.")
    k.observe(5, "37?")
    b, notes = k.block(6)
    assert b == ("(stream notes: game: number guess 1-50, her secret is 37, "
                 "guesses: 25 too small, 40 too big, 37 correct)")
    assert notes.game and "37" in notes.game[0]
    assert k.block(7)[0] == ""                       # the game is over
    # it renders with the live renderer and fits its budget
    assert len(b) <= mem.CHARS
    assert dn.guess_of("I pick 7 or maybe 99?") == 7
    assert dn.guess_of("no digits") is None and dn.guess_of("it is 500") is None


def test_other_games_corrections_and_names():
    k = dn.Keeper(dn.Plan(kind="chain", start=2, length=5, secret=None))
    for t, w in ((3, "apple"), (4, "Egg"), (5, "that is a long sentence now")):
        k.observe(t, w)
    assert "used: apple, egg" in k.block(6)[0]
    q = dn.Keeper(dn.Plan(kind="twenty", start=2, length=6, secret="a kite"))
    for t in (3, 4, 5):
        q.observe(t, "Is it alive?")
    assert "her secret is a kite, 3 of 20 questions asked" in q.block(6)[0]
    c = dn.Keeper()
    c.observe(2, "It is not Kyoto, it is Kobe.", "I love Kyoto.")
    c.observe(3, "Hi! My name is Mika.", "")
    b, n = c.block(4)
    assert n.corrections == ["Kobe, never Kyoto"]
    assert n.facts == ["viewer's name is Mika"]
    assert b == ("(stream notes: corrections: Kobe, never Kyoto; facts: "
                 "viewer's name is Mika)")


def test_plan_game_is_seeded_and_needs_notes_for_secrets():
    a = dn.plan_game(123, 8, 1.0, True)
    assert a == dn.plan_game(123, 8, 1.0, True) and a is not None
    assert 2 <= a["start"] <= 3 and a.end <= 8 + 1
    assert dn.plan_game(123, 5, 1.0, True) is None        # too short
    assert dn.plan_game(123, 8, 0.0, True) is None
    kinds = {dn.plan_game(s, 8, 1.0, False)["kind"] for s in range(60)}
    assert kinds == {"chain", "rather"}
    assert {dn.plan_game(s, 8, 1.0, True)["kind"]
            for s in range(60)} == set(dn.GAMES)
    n = sum(dn.plan_game(s, 8, 0.3, True) is not None for s in range(400))
    assert 80 < n < 160
    assert dn.lang_for(dn.Plan(kind="chain", start=2, length=4, secret=None),
                       2) == "en"


# ----------------------------------------------------- tail block (reminders)

def test_the_dataset_tail_is_the_live_tail():
    """flow.live's Performer.tail_block and the data generator build the
    same block from the same inputs."""
    from flow import live
    bm = _body("Hiyori")
    can = rl.capabilities(bm)
    notes = mem.Notes(game=["number guess 1-50, her secret is 37"])
    nb = notes.render()
    fake = SimpleNamespace(
        memory=SimpleNamespace(block=lambda: nb, current=lambda: notes),
        reminders="auto", body={"body": bm}, rules_ids=[], _rule_can=can,
        emit=lambda s: None)
    for line, prev in (("Look to your left!", "Hello."),
                       ("Is it 25? And dance for me!", "Ok [pose {}] sure."),
                       ("hello", "")):
        live_block = live.Performer.tail_block(fake, line, prev, False, 3)
        got, ids = rl.tail_block(line, prev, nb, notes, can, "auto")
        assert got == live_block and ids == fake.rules_ids
    assert "(reminders: " in got or got == nb
    assert rl.tail_block("x", "", nb, notes, can, "off") == (nb, [])
    assert rl.tail_block("x", "", "", None, can, "off") == ("", [])


def test_beats3_is_selectable_wherever_beats2_is():
    from flow import director as dr
    from flow.train import evalsft, sft
    ap = dr.argparse.ArgumentParser()
    dr.add_args(ap)
    for p in ("beats3", "short+bible"):
        assert ap.parse_args(["--body-prompt", p]).body_prompt == p
    assert ds.parse_opts(["gen", "--prompt", "beats3"]).teacher_prompt \
        == "beats3"
    assert "short+bible" in dx.STUDENT_PROMPTS
    # the director builds the prompt the data generator wrote for the same
    # persona text (bible behind its separator), a body file and no rig
    bm = _body("Hiyori")
    persona = PERSONA + pz.BIBLE_SEP + "- likes tea"
    a = ap.parse_args(["--mode", "body", "--body-prompt", "beats3",
                       "--persona", persona])
    ml, _names = dr.move_menu(a, bm)
    assert pz.system_prompt(bm, a.persona, False, "", ml, a.body_prompt) \
        == pz.system_prompt(bm, persona, False, "", ml, "beats3")
    # evalsft and the trainer take the student prompts
    assert sft.parse(["stats", "--data", "x", "--student-prompt",
                      "short+bible"]).student_prompt == "short+bible"
    o = evalsft.parse(["run", "--data", "x", "--out", "y", "--split", "s",
                       "--name", "n",
                       "--student-prompt", "short+bible"])
    assert o.student_prompt == "short+bible"
