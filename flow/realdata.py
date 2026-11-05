#!/usr/bin/env python3
"""
Real streamer chat as style references for the data generator, read from
files the user downloaded (nothing is fetched or crawled here, and no file
of this kind is in the repository). Licences and what each dataset allows:
docs/research.md, "Real streamer data". A dataset whose licence is not
confirmed there ("do not use until confirmed") must not be passed to the
flags below.

Two loaders, both deterministic for a seed:

  viewer_lines(path, lang, n, seed)
      real chat lines (viewers' messages), cleaned, for flow.dataset
      --real-viewer: style references and openers for the simulated viewer.
  streamer_examples(path, n, seed)
      short anonymised (context, streamer reply) pairs for flow.dataset
      --style-seeds: rhythm examples for the beats3 teacher.

Formats read (by suffix, then by content):

  .jsonl / .ndjson   one JSON object a line. The viewer text is the first of
                     message, text, content, body, msg, comment,
                     audience_comment; "lang" / "language" (en, ja, zh) is
                     read when present, else detected from the script.
                     Author / user / channel / id fields are never read.
                     A pair needs "context" (or audience_comment, comment)
                     and "reply" (or streamer_response, response).
  .json              {"streamer1": [[audience, response], ...], ...} (the
                     Subset/subset.json of LiveChat), a list of
                     [streamer_id, audience, response] lists (LiveChat's
                     processed data), or a list of the objects above.
  .pk / .pkl         LiveChat's ProcessedData files: pickled lists of
                     [streamer_id, audience_comment, streamer_response].
                     Read through a restricted unpickler that only builds
                     lists, tuples, dicts, strings and numbers; anything
                     else raises. Still, only open files from the source.
  .txt               one chat line a line.

Cleaning (clean_line, also applied to the reply side of a pair), for a line
to be used:
  - NFKC-normalised, whitespace collapsed; a leading "name:" or "[name]"
    (a chat export's username) is removed, and "@mentions" are removed;
  - dropped when it still has a URL, an e-mail address, a run of 7 or more
    digits (phone numbers, ids), or a leading/trailing handle;
  - dropped when it is emote-only (":emote:" tokens, emoji, "w", "草",
    "8888", "哈哈哈哈", kaomoji-like symbol runs, fewer than 3 letters),
    spam (one character or word repeated for most of the line, or more than
    4 identical characters in a row), or too long (> MAX_CHARS, 120);
  - dropped when it matches the keyword filter BLOCK_EN / BLOCK_CJK: slurs
    and abuse, sexual words, self-harm and violence, in English, Japanese
    and Chinese. It is deliberately about toxicity, not politics (a
    political line is a topic, not abuse: no political name is on it) and
    not about mild playful words ("ばか", "垃圾", "idiot"-like banter is
    not all on it; the English list is word-bounded, so "cocktail" and
    "Sussex" survive). A keyword list is crude: it misses misspellings and
    context and still drops a few innocent lines; it is a first filter, and
    the pilot's lines are to be read by a person before a full run
    (scripts/style_ref.py --show 50);
  - for pairs: gift / follow thank-yous and welcomes, which carry a
    username ("感谢XX的火箭", "welcome X to the stream"), are dropped, and
    so is a streamer name that appears in the pair as a mention.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import random
import re
import unicodedata

MAX_CHARS = 120
MIN_CHARS = 6
MIN_CJK = 3
VIEWER_KEYS = ("message", "text", "content", "body", "msg", "comment",
               "audience_comment")
CONTEXT_KEYS = ("context", "audience_comment", "comment", "message")
REPLY_KEYS = ("reply", "streamer_response", "response")

URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|jp|cn|tv|io|gg)"
                 r"\b", re.I)
EMAIL = re.compile(r"\S+@\S+\.\S+")
LONG_DIGITS = re.compile(r"\d{7,}|\d[\d -]{8,}\d")
MENTION = re.compile(r"@\S+")
HANDLE_PREFIX = re.compile(r"^\s*(?:\[[^\]]{1,24}\]|<[^>]{1,24}>|"
                           r"[\w.\-぀-鿿]{1,20}\s?[:：])\s*")
EMOTE_TOKEN = re.compile(r":[\w\-]{2,32}:|\[[A-Za-z0-9_一-鿿]{1,8}\]"
                         r"|\(\([^)]*\)\)")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿"
                   "️‍]")
LAUGH = re.compile(r"^(?:w+|草+|ｗ+|8{3,}|6{3,}|哈+|www+|lol+|lmao+|kek+|"
                   r"haha+|ahaha+|ー|[!?！？.。、,~]+)$", re.I)
GIFT_THANKS = re.compile(
    r"感谢.{0,16}(?:礼物|火箭|嘉年华|飞机|跑车|关注|点赞|守护|舰长|打赏|"
    r"的)|谢谢.{0,16}(?:礼物|火箭|关注|点赞|守护|舰长)|欢迎.{0,16}(?:进入|来到|"
    r"进来|直播间)|(?:thanks?|thank you|welcome)\s+(?:to\s+)?@?[A-Z]\w+\s*"
    r"(?:for|to the)|ありがとう.{0,12}(?:ギフト|スパチャ|フォロー)|"
    r"(?:ようこそ|いらっしゃい).{0,12}さん", re.I)

# The keyword filter (crude, see the module docstring): lower-cased
# substrings for the CJK entries, word-bounded regexes for English.
BLOCK_EN = re.compile(
    r"\b(?:fuck\w*|shit(?:ty|head|bag)?|bitch\w*|cunt\w*|dick(?:head)?|"
    r"pussy|slut\w*|whore\w*|nigg\w*|fag(?:s|got\w*)?|retard\w*|rap(?:e|ed|"
    r"ing|ist)|nude\w*|nsfw|porn\w*|sex(?:y|ual|ually|ting|ts)?|horny|"
    r"boobs?|titties|asshole|cumming|dildo|blowjob|handjob|orgasm\w*|kys|"
    r"kill\s+yourself|suicid\w*|bastard|onlyfans|lewd|hentai|ecchi|thot|"
    r"incest|molest\w*|pedo\w*|loli(?:con)?|shota|bomb\s+threat)\b", re.I)
BLOCK_CJK = ("死ね", "しね", "殺す", "ころす", "きもい", "キモい", "キモ", "ブス",
             "デブ", "ハゲ", "ちんこ", "ちんぽ", "まんこ", "おっぱい", "エロ",
             "セックス", "オナニー", "レイプ", "犯す", "自殺", "自傷", "ロリ",
             "ショタ", "ホモ", "ガイジ", "池沼", "障害者", "チョン", "シナ人",
             "売春", "傻逼", "傻b", "沙比", "煞笔", "妈的", "媽的", "你妈",
             "你媽", "他妈", "操你", "草你", "尼玛", "去死", "杀了你", "殺了你",
             "自杀", "自殺", "弱智", "脑残", "腦殘", "智障", "婊子", "鸡巴",
             "雞巴", "色情", "裸体", "裸體", "裸照", "做爱", "做愛", "约炮",
             "約炮", "强奸", "強姦", "性爱", "性愛", "支那", "黑鬼", "日本鬼子",
             "小日本", "棒子", "死妈", "死媽")


class UnsafePickle(Exception):
    pass


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        raise UnsafePickle(f"{module}.{name}: the file builds more than "
                           "lists, tuples, dicts, strings and numbers")


def _load_pickle(path):
    with open(path, "rb") as fh:
        return _Restricted(io.BytesIO(fh.read())).load()


def _lang_of(text):
    try:
        from . import style as sty
    except ImportError:
        from flow import style as sty
    return sty.lang_of(text)


def blocked(text) -> bool:
    """The keyword filter: toxic, abusive, sexual or violent wording."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    return bool(BLOCK_EN.search(t)) or any(k.lower() in t for k in BLOCK_CJK)


def _emote_only(s) -> bool:
    rest = EMOTE_TOKEN.sub(" ", s)
    rest = EMOJI.sub(" ", rest).strip()
    if not rest or LAUGH.match(rest):
        return True
    letters = sum(1 for c in rest if c.isalpha())
    return letters < 3


def _spam(s) -> bool:
    if re.search(r"(.)\1{4,}", s):
        return True
    ws = s.split()
    if len(ws) >= 4 and len(set(w.lower() for w in ws)) <= len(ws) // 3:
        return True
    chars = [c for c in s if not c.isspace()]
    return len(chars) >= 8 and len(set(chars)) <= 3


def clean_line(text, max_chars=MAX_CHARS):
    """The line cleaned for use, or None when it cannot be used (module
    docstring). The tests (URL, digits, keywords) read the NFKC form of the
    line, the text kept is the NFC form, so a full-width ！ or 、 stays."""
    if not isinstance(text, str):
        return None
    raw = unicodedata.normalize("NFC", text).replace("\u200b", "")
    raw = re.sub(r"\s+", " ", raw).strip()
    k = unicodedata.normalize("NFKC", raw)
    if URL.search(k) or EMAIL.search(k) or LONG_DIGITS.search(k):
        return None
    s = HANDLE_PREFIX.sub("", raw, count=1) if HANDLE_PREFIX.match(raw) \
        and len(raw) > 12 else raw
    s = MENTION.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-")
    cjk = any(ord(c) >= 0x3040 for c in s)
    if not s or len(s) > max_chars or len(s) < (MIN_CJK if cjk
                                                else MIN_CHARS):
        return None
    k = unicodedata.normalize("NFKC", s)
    if _emote_only(k) or _spam(k) or blocked(k):
        return None
    return s


def _flatten(data):
    """Yield dicts with "viewer" / "reply" / "lang" from any loaded
    structure."""
    if isinstance(data, dict):
        # LiveChat subset: {"streamer1": [[audience, response], ...]}
        if data and all(isinstance(v, list) for v in data.values()) \
                and not any(k in data for k in VIEWER_KEYS):
            for v in data.values():
                yield from _flatten(v)
            return
        v = next((data[k] for k in CONTEXT_KEYS if isinstance(
            data.get(k), str)), None)
        for k in VIEWER_KEYS:
            if isinstance(data.get(k), str):
                v = data[k]
                break
        r = next((data[k] for k in REPLY_KEYS if isinstance(
            data.get(k), str)), None)
        lang = data.get("lang") or data.get("language")
        yield {"viewer": v, "reply": r,
               "lang": lang if isinstance(lang, str) else None}
    elif isinstance(data, (list, tuple)):
        if len(data) == 3 and all(isinstance(x, (str, int)) for x in data) \
                and isinstance(data[1], str) and isinstance(data[2], str):
            yield {"viewer": data[1], "reply": data[2], "lang": None}
        elif len(data) == 2 and all(isinstance(x, str) for x in data):
            yield {"viewer": data[0], "reply": data[1], "lang": None}
        else:
            for x in data:
                yield from _flatten(x)
    elif isinstance(data, str):
        yield {"viewer": data, "reply": None, "lang": None}


def read_rows(path):
    """Every {"viewer", "reply", "lang"} of a file (formats: module
    docstring), lazily for .jsonl and .txt."""
    low = path.lower()
    if low.endswith((".jsonl", ".ndjson")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                yield from _flatten(obj)
    elif low.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            yield from _flatten(json.load(fh))
    elif low.endswith((".pk", ".pkl", ".pickle")):
        yield from _flatten(_load_pickle(path))
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield {"viewer": line.strip(), "reply": None,
                           "lang": None}


def _pool(path, field, lang=None, cap=200000):
    """Cleaned, deduplicated lines of one side, in file order (at most
    `cap` read), filtered to a language when asked."""
    seen, out = set(), []
    for row in read_rows(path):
        raw = row.get(field)
        s = clean_line(raw)
        if s is None:
            continue
        key = re.sub(r"\W+", "", s.lower())
        if key in seen:
            continue
        if lang:
            lg = row.get("lang") or _lang_of(s)
            if lg != lang:
                continue
        seen.add(key)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def viewer_lines(path, lang, n, seed=0) -> list:
    """Up to n cleaned real chat lines in `lang` ("en", "ja", "zh"), a
    seeded sample of the file (the first 200000 usable lines)."""
    pool = _pool(path, "viewer", lang)
    rng = random.Random(f"{seed}:viewer:{lang}")
    rng.shuffle(pool)
    return pool[:max(0, int(n))]


def streamer_examples(path, n, seed=0) -> list:
    """Up to n [{"context", "reply", "lang"}], anonymised (cleaned on both
    sides, gift thank-yous and welcomes dropped, the reply 8 to 120
    characters): rhythm seeds, a seeded sample."""
    pairs, seen = [], set()
    for row in read_rows(path):
        if not row.get("reply") or not row.get("viewer"):
            continue
        if GIFT_THANKS.search(row["reply"]) or GIFT_THANKS.search(
                row["viewer"]):
            continue
        c, r = clean_line(row["viewer"]), clean_line(row["reply"])
        if c is None or r is None or len(r) < 8:
            continue
        key = re.sub(r"\W+", "", r.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"context": c, "reply": r,
                      "lang": row.get("lang") or _lang_of(r)})
        if len(pairs) >= 200000:
            break
    rng = random.Random(f"{seed}:seeds")
    rng.shuffle(pairs)
    return pairs[:max(0, int(n))]


# ------------------------------------------------- licences and attribution

# Recorded in docs/research.md ("Real streamer data"). `status` says whether
# the licence has been read from the source: only LiveChat's repository was
# readable when this was written; the two Hugging Face cards were not.
SOURCES = {
    "livechat": {
        "name": "LiveChat (Gao et al., ACL 2023)",
        "url": "https://github.com/gaojingsheng/LiveChat",
        "status": "code is MIT; the data's own terms are not stated",
        "confirmed": False,
        "attribution": "Contains data from LiveChat (Gao, Lian, Zhou, Fu, "
                       "Wang: LiveChat: A Large-Scale Personalized Dialogue "
                       "Dataset Automatically Constructed from Live "
                       "Streaming, ACL 2023), https://github.com/"
                       "gaojingsheng/LiveChat.",
    },
    "oshichats": {
        "name": "OshiChats v2 (pykeio)",
        "url": "https://huggingface.co/datasets/pykeio/oshichats-v2",
        "status": "reported CC BY 4.0, not read from the card",
        "confirmed": False,
        "attribution": "Contains data from OshiChats v2 by pyke.io "
                       "(https://huggingface.co/datasets/pykeio/"
                       "oshichats-v2), licensed CC BY 4.0 as reported "
                       "(confirm on the card).",
    },
    "processed_chats": {
        "name": "Differentiable/processed_chats",
        "url": "https://huggingface.co/datasets/Differentiable/"
               "processed_chats",
        "status": "reported bigscience-openrail-m, not read from the card",
        "confirmed": False,
        "attribution": "Contains data from Differentiable/processed_chats "
                       "(bigscience-openrail-m as reported; confirm on the "
                       "card).",
    },
    "custom": {
        "name": "a file of the user's own",
        "url": None,
        "status": "the user's responsibility",
        "confirmed": True,
        "attribution": "",
    },
}


def source_info(source) -> dict:
    if source not in SOURCES:
        raise SystemExit(f"--real-source {source!r}: one of "
                         f"{', '.join(SOURCES)}")
    return {"id": source, **SOURCES[source]}


def guess_source(path) -> str:
    p = os.path.basename(path).lower() + " " + os.path.dirname(
        os.path.abspath(path)).lower()
    for key, hint in (("livechat", "livechat"), ("oshichats", "oshi"),
                      ("processed_chats", "processed_chats")):
        if hint in p:
            return key
    return "custom"


def style_records(path, side="auto", n=0, seed=0, clean=True) -> list:
    """Records for flow.style.style from a real file: {"spoken", "user",
    "session", "turn"}. side "reply": the streamer's reply is `spoken` and
    the comment before it `user` (pairs: LiveChat); "viewer": the chat line
    is `spoken` (a file of chat lines only); "auto": reply when the file has
    pairs. Each record is its own session (no earlier turns, so the
    callback number is only its cues). clean=False keeps the lines as
    they are (the filters of clean_line drop what a streamer's file has
    too: short interjections, some long replies); n > 0 takes a seeded
    sample."""
    rows = list(read_rows(path))
    if side == "auto":
        side = "reply" if any(r.get("reply") for r in rows) else "viewer"
    out = []
    for i, r in enumerate(rows):
        if side == "reply":
            if not r.get("reply"):
                continue
            spoken, user = r["reply"], r.get("viewer") or ""
        else:
            spoken, user = r.get("viewer"), ""
        if not isinstance(spoken, str):
            continue
        if clean:
            spoken = clean_line(spoken, max_chars=400)
            user = clean_line(user, max_chars=400) or ""
        else:
            spoken = spoken.strip()
        if not spoken:
            continue
        out.append({"spoken": spoken, "user": user, "session": f"r{i}",
                    "turn": 1})
    if n and len(out) > n:
        out = random.Random(f"{seed}:style").sample(out, n)
    return out
