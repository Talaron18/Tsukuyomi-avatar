"""
flow.realdata (loaders and filters for real streamer chat) and the flags that
use it in flow.dataset (--real-viewer, --style-seeds), with local fake files
and fake models: no network, no real dataset.
"""

from __future__ import annotations

import json
import os
import pickle

import pytest

from flow import dataexport as dx
from flow import pose as pz
from flow import realdata as rd
from tests.test_beats3_data import _gen

# ------------------------------------------------------------- real data

VIEWER = [
    {"message": "that boss fight was insane tonight", "author": "BigFan99",
     "lang": "en"},
    {"message": "can you play the piano part again please", "id": "u1"},
    {"message": "lol", "author": "x"},
    {"message": ":emote1: :emote2:"},
    {"message": "check out https://spam.example.com now please"},
    {"message": "call me at 555-123-4567 tomorrow ok"},
    {"message": "aaaaaaaaaaaaaaaaaaaaaaaaaa"},
    {"message": "fuck this stream honestly"},
    {"message": "@Streamer you are so funny today honestly"},
    {"message": "x" * 200},
    {"message": "I love how calm this song is today"},
    {"message": "今日の配信すごく楽しかったです！"},
    {"message": "主播唱得真好听，再来一首吧"},
    {"text": "that boss fight was insane tonight"},
]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in rows), encoding="utf-8")
    return str(path)


def test_viewer_lines_are_cleaned_and_filtered(tmp_path):
    p = _write_jsonl(tmp_path / "chat.jsonl", VIEWER)
    en = rd.viewer_lines(p, "en", 20, 1)
    assert sorted(en) == sorted([
        "that boss fight was insane tonight",
        "can you play the piano part again please",
        "you are so funny today honestly",
        "I love how calm this song is today"])
    assert rd.viewer_lines(p, "ja", 5, 1) == ["今日の配信すごく楽しかったです！"]
    assert rd.viewer_lines(p, "zh", 5, 1) == ["主播唱得真好听，再来一首吧"]
    assert rd.viewer_lines(p, "en", 2, 1) == rd.viewer_lines(p, "en", 2, 1)
    assert len(rd.viewer_lines(p, "en", 2, 1)) == 2
    for s in en:
        assert len(s) <= rd.MAX_CHARS and "@" not in s


def test_cleaning_rules():
    c = rd.clean_line
    assert c("BigFan99: that boss fight was insane tonight") \
        == "that boss fight was insane tonight"
    assert c("w w w w w w w") is None and c("草草草草草草") is None
    assert c("www.example.org is great today") is None
    assert c("mail me a@b.com please do") is None
    assert c("ok") is None and c("😂😂😂😂😂😂😂") is None
    assert c("buy buy buy buy buy buy buy buy") is None
    assert c("死ね、このゲームは最悪だ") is None
    assert c("你这个傻逼主播真的很烦") is None
    assert c("Nice cover of that song, loved it") == \
        "Nice cover of that song, loved it"


def test_livechat_formats_and_pairs(tmp_path):
    sub = {"streamer1": [["我的用了4年了", "用了4年是手机吗？大家先帮主播点点赞好不好？"],
                         ["说我又不结婚装修干嘛", "准备什么时候装修啊，我也想知道呢"],
                         ["谢谢", "感谢小明的火箭，欢迎大家来到直播间啊"],
                         ["大家好啊主播", "你好你好，今天想听什么歌呢，点歌的随便点"]],
           "streamer2": [["听不清楚", "我调一下麦克风，稍等一下哦各位"]]}
    j = tmp_path / "subset.json"
    j.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    pk = tmp_path / "train_data.pk"
    with open(pk, "wb") as fh:
        pickle.dump([["streamer1", "今天唱什么歌呀主播", "唱一首慢一点的吧，大家安静听"],
                     ["streamer2", "主播吃饭了吗", "还没呢，等一下直播完再去吃东西"]], fh)
    for path in (str(j), str(pk)):
        ps = rd.streamer_examples(path, 10, 0)
        assert ps and all(set(p) == {"context", "reply", "lang"} for p in ps)
        assert all(p["lang"] == "zh" and 8 <= len(p["reply"]) <= 120
                   for p in ps)
        assert not any("感谢" in p["reply"] or "火箭" in p["reply"]
                       for p in ps)           # a gift thank-you has a name
    assert len(rd.streamer_examples(str(j), 10, 0)) == 4
    assert rd.streamer_examples(str(j), 2, 5) == rd.streamer_examples(
        str(j), 2, 5)
    zh = rd.viewer_lines(str(j), "zh", 10, 0)
    assert "我的用了4年了" in zh and "谢谢" not in zh


def test_a_pickle_that_builds_objects_is_refused(tmp_path):
    p = tmp_path / "evil.pk"
    with open(p, "wb") as fh:
        pickle.dump({"a": os.getcwd}, fh)
    with pytest.raises(rd.UnsafePickle):
        list(rd.read_rows(str(p)))


def test_real_viewer_and_style_seeds_in_the_data(tmp_path):
    chat = _write_jsonl(tmp_path / "chat.jsonl", [
        {"message": "that boss fight was insane tonight"},
        {"message": "can you play the piano part again please"},
        {"message": "I love how calm this song is today"},
        {"message": "what is your favourite snack at night"}])
    seeds = tmp_path / "subset.json"
    seeds.write_text(json.dumps({"s": [
        ["今天唱什么歌呀主播", "唱一首慢一点的吧，大家安静听"],
        ["主播吃饭了吗", "还没呢，等一下直播完再去吃东西"],
        ["听不清楚", "我调一下麦克风，稍等一下哦各位"],
        ["晚上好主播", "晚上好呀，今天过得怎么样，有什么新鲜事"]]},
        ensure_ascii=False), encoding="utf-8")
    base = ["--real-viewer", chat, "--style-seeds", str(seeds),
            "--real-source", "custom", "--real-verbatim", "1",
            "--game-share", "0", "--langs-unused"]
    base.pop()
    out, recs, made = _gen(tmp_path, *base, sessions=2, turns=5,
                           avatars="Hiyori")
    # without --real-licence-ok a dataset of a named source is refused
    with pytest.raises(SystemExit, match="licence"):
        _gen(tmp_path / "r2", "--real-viewer", chat, "--real-source",
             "livechat", sessions=1, turns=2)
    reals = [r for r in recs if r.get("real")]
    assert len(reals) == len(recs)
    first = [r for r in recs if r["turn"] == 1]
    for r in first:
        assert r["real"]["seeds"] and len(r["real"]["seeds"]) == 4
    # verbatim lines: used as the user line, recorded, and from the file
    # (an English session; others find no line in their language)
    verb = [r for r in recs if r["real"].get("verbatim")]
    for r in verb:
        assert r["user"] == r["real"]["verbatim"]
    # the character's system prompt got the seeds, as rhythm only; the
    # record's prompt is the base one
    reg = dx.load_registry(out)
    for b in made["character"]:
        for system, _m in b.sent:
            assert "for rhythm only" in system and "How real streamers" in system
            assert system.split(pz.__dict__["SEEDS_HEAD"] if False else
                                "\n\nHow real streamers")[0] in \
                reg["prompts"].values()
    # the simulator got style references
    assert any("only for the way people type" in c for b in made["user"]
               for _s, c in b.sent) or not verb
    # the attribution travels with the data
    avatars = json.load(open(os.path.join(out, "avatars.json")))
    assert avatars["sources"]["custom"]["id"] == "custom"
    assert os.path.exists(os.path.join(out, "README.md"))


def test_licence_status_is_recorded_for_the_named_sources():
    for k in ("livechat", "oshichats", "processed_chats"):
        info = rd.source_info(k)
        assert info["confirmed"] is False and info["status"]
    assert "pyke.io" in rd.SOURCES["oshichats"]["attribution"]
    assert rd.guess_source("/data/oshichats-v2/chats.jsonl") == "oshichats"
    assert rd.guess_source("/x/LiveChat/Dataset/Subset/subset.json") \
        == "livechat"
    assert rd.guess_source("/x/mine.jsonl") == "custom"
    with pytest.raises(SystemExit):
        rd.source_info("nope")


def test_the_keyword_filter_is_about_toxicity_and_spares_innocent_lines():
    for ok in ("黄色的花很好看呀", "a cocktail bar downtown tonight",
               "我是中国人，来自上海", "the tits on the feeder were cute",
               "Sussex is lovely in spring ok", "cum laude graduate today",
               "今日はばかみたいに暑いね", "习近平 is in the news today",
               "那个垃圾袋放在门口了"):
        assert not rd.blocked(ok), ok
        assert rd.clean_line(ok) == ok or len(ok) < rd.MIN_CHARS, ok
    for bad in ("fuck this stream honestly", "你这个傻逼主播真的很烦",
                "死ね、このゲームは最悪だ", "just go kill yourself already",
                "send nudes please lol"):
        assert rd.blocked(bad), bad
        assert rd.clean_line(bad) is None, bad
