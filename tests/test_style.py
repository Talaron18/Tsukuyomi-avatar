"""
flow.style: the style numbers (words and sentences a reply, distinct-n, echo,
question endings, openings, interjections, substance) on small hand-made
replies, in the checks table, and on the held-out set of q8r3 (the table of
docs/embodiment.md, "Style checks").
"""

from __future__ import annotations

import json
import os

import pytest

from flow import checks as ck
from flow import style as sty

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "runs", "eval_q8r3")


def rec(spoken, user="", session="s", turn=1):
    return {"spoken": spoken, "user": user, "session": session, "turn": turn}


def test_language_words_and_sentences():
    assert sty.lang_of("Oh, I really did not expect that.") == "en"
    assert sty.lang_of("えっ、本当ですか？") == "ja"
    assert sty.lang_of("真的吗？我不信。") == "zh"
    assert sty.lang_of("OK!") is None
    assert sty.words("Don't you dare, well-known cat!") == [
        "Don't", "you", "dare", "well-known", "cat"]
    assert len(sty.sentences("One. Two! Three?... Four")) == 4
    assert len(sty.sentences("...")) == 0


def test_numbers_by_language():
    st = sty.style([rec("One day I saw a cat. Then another one."),
                    rec("I have two cats and a dog today, really."),
                    rec("今日は3匹の猫を見たよ。"),
                    rec("昨天我吃了三个苹果，真的。")])
    assert st["n"] == 4
    assert st["lang"] == {"en": 2, "ja": 1, "zh": 1}
    assert st["words_en"] == pytest.approx((9 + 9) / 2)
    assert st["sentences"] == pytest.approx((2 + 1 + 1 + 1) / 4)
    assert st["chars_zh"] == len("昨天我吃了三个苹果，真的。")
    # the substance kinds: "One day" is not a number, "two cats" and 3匹 are
    assert sty.has_number("I have two cats") and sty.has_number("今日は3匹")
    assert sty.has_number("三个苹果") and not sty.has_number("One day, hm.")


def test_distinct_and_unique_words():
    st = sty.style([rec("the cat sat on the mat today ok"),
                    rec("the cat sat on the mat today ok")])
    assert st["distinct1"] == pytest.approx(7 / 16)
    assert st["distinct2"] == pytest.approx(7 / 14)
    assert st["unique_words_en"] == 7
    fresh = sty.style([rec("one two three four five six"),
                       rec("seven eight nine ten eleven twelve")])
    assert fresh["distinct1"] == fresh["distinct2"] == 1.0


def test_echo_needs_two_content_words_or_a_cjk_substring():
    u = "What is your favorite pizza topping, anyway?"
    assert sty.echoes("My favorite pizza is plain.", u)
    assert not sty.echoes("Pineapple, obviously, do not judge me.", u)
    assert not sty.echoes("Pizza is fine.", u)                # one word
    assert sty.echoes("好きなピザのトッピングは、チーズだよ。",
                      "好きなピザのトッピングは何？")
    assert not sty.echoes("チーズだよ。", "好きなピザのトッピングは何？")
    st = sty.style([rec("My favorite pizza is plain. Boring, yes.", u),
                    rec("Pineapple. Obviously.", u),
                    rec("Anything at all.", "(chat is quiet)")])
    assert st["echo"] == pytest.approx(0.5)       # the quiet turn is left out


def test_question_openings_interjections():
    st = sty.style([rec("Oh, really? Tell me more?"),
                    rec("Hmm. I think so."),
                    rec("えっ、そうなの？"),
                    rec("Sure thing, go ahead."),
                    rec("嗯，我觉得可以。")])
    assert st["question"] == pytest.approx(2 / 5)
    assert st["interjection"] == pytest.approx(4 / 5)
    assert sty.opening("Hmm, well.") == "hmm"
    assert sty.opening("えっ、そうなの") == "えっ"
    assert sty.opening("“Wow!” she said") == "wow"
    assert dict(st["top_openings"]).keys() >= {"oh"}
    assert not sty.interjection("Sure thing.") \
        and not sty.interjection("Honestly, okay.")
    assert sty.ends_question("Really?”") and not sty.ends_question("No.")


def test_substance_kinds():
    k = sty.substance_kinds
    assert k("Oh!", "hi") == []
    assert "number" in k("It took me 3 hours.")
    assert "named" in k("We went to Kyoto last year.")
    assert "named" in k("今日はポケモンやってたよ。")
    assert "named" in k("「君の名は」が好き。")
    assert "named" not in k("Well, that is nice.")
    assert "opinion" in k("I think tea beats coffee.")
    assert "opinion" in k("When I was little I hid in a box.")
    assert "opinion" in k("私は雨の日が好きだな。")
    assert "opinion" in k("我觉得这个不错。")
    assert "opinion" not in k("That is nice of you.")
    # a callback: a cue, or two words of an earlier line the current lacks
    assert "callback" in k("Like you said earlier, yes.")
    assert "callback" in k("さっきの話、覚えてるよ。")
    assert "callback" in k("刚才你说的那个。")
    assert "callback" in k("How is your sourdough bread doing?",
                           "How are you?",
                           ["I started baking sourdough bread today"])
    assert "callback" not in k("How are you?", "How are you?",
                               ["I started baking sourdough bread today"])


def test_callback_uses_the_sessions_earlier_user_lines():
    recs = [rec("Sure.", "I adopted a grey kitten named Mochi", "a", 1),
            rec("Fine.", "anyway what is for dinner", "a", 2),
            rec("Did Mochi settle in, the grey kitten?", "ok", "a", 3)]
    st = sty.style(recs)
    assert st["substance_kinds"]["callback"] == pytest.approx(1 / 3)
    # the first turn of another session does not see session a's lines
    st2 = sty.style([rec("Did Mochi settle in, the grey kitten?", "ok", "b", 1)])
    assert st2["substance_kinds"]["callback"] == 0


def test_empty_and_unspoken_records():
    assert sty.style([]) == {"n": 0}
    assert sty.style([rec("  ")]) == {"n": 0}
    assert sty.rows_of({"n": 0}) == []


def test_style_rows_are_in_the_checks_table(tmp_path):
    rows = [{"id": f"s/t{i:02d}", "session": "s", "turn": i, "avatar": "Hiyori",
             "user": "Hello there my friend", "reply_raw": f"Oh, wow {i}! Hi.",
             "spoken": f"Oh, wow {i}! Hi."} for i in range(1, 4)]
    p = tmp_path / "student.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    [(label, recs)] = ck.load(str(p))
    res = ck.run(recs, ck.Bodies(None, os.path.join(ROOT, "rigs")))
    assert res["style"]["n"] == 3
    text = ck.tables([label], [res])
    for want in ("style: replies", "words/reply (en)", "distinct-2 (en)",
                 "ends with a question", "substance (any)"):
        assert want in text
    # the failure checks come first, the style rows after them
    assert text.index("quiet_says_quiet") < text.index("style: replies")
    # an input without style numbers shows "-"
    t2 = ck.tables(["a", "b"], [res, {"n": 0, "checks": res["checks"]}])
    assert "style: replies" in t2


@pytest.mark.skipif(not os.path.exists(os.path.join(EVAL, "student.jsonl")),
                    reason="runs/eval_q8r3 is not in this checkout")
def test_q8r3_eval_reproduces_the_style_table():
    """The numbers of docs/embodiment.md ("Style checks") on the 820 held-out
    turns of q8r3 and its teacher. The first comparison's figures (78 / 72
    chars, 24.7 / 22.8 English words, echo 28% / 38%, questions 24% / 21%)
    come out of these definitions; the others are the documented ones."""
    def load(name):
        out = []
        for line in open(os.path.join(EVAL, name), encoding="utf-8"):
            r = json.loads(line)
            out.append({k: r[k] for k in ("spoken", "user", "session",
                                          "turn")})
        return out
    t, s = sty.style(load("recorded.jsonl")), sty.style(load("student.jsonl"))
    assert t["n"] == s["n"] == 820
    assert t["chars"] == pytest.approx(78.3, abs=0.2)
    assert s["chars"] == pytest.approx(71.9, abs=0.2)
    assert t["words_en"] == pytest.approx(24.6, abs=0.3)
    assert s["words_en"] == pytest.approx(22.8, abs=0.3)
    assert t["echo"] == pytest.approx(0.28, abs=0.01)
    assert s["echo"] == pytest.approx(0.38, abs=0.01)
    assert t["question"] == pytest.approx(0.24, abs=0.01)
    assert s["question"] == pytest.approx(0.21, abs=0.01)
    assert t["unique_words_en"] == pytest.approx(1268, abs=20)
    assert s["unique_words_en"] == pytest.approx(1023, abs=20)
    assert t["distinct2"] == pytest.approx(0.75, abs=0.02)
    assert s["distinct2"] == pytest.approx(0.70, abs=0.02)
    # the student is the less varied: fewer unique words, more echo
    assert s["unique_words_en"] < t["unique_words_en"]
    assert s["interjection"] > t["interjection"]
