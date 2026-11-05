#!/usr/bin/env python3
"""
Style numbers for a set of replies: how long, how varied, how much substance.
Numbers, not pass / fail checks: they are shown under the failure checks in
scripts/checks.py's table (flow.checks) and by scripts/style_ref.py on a file
of real streamer replies, so a teacher prompt can be tuned until it sits near
real streamers (docs/embodiment.md, "Style checks").

The input is a list of records, dicts with "spoken" (the reply without its
markers), "user" (the line it answers), "session" and "turn" (the order inside
a session, for the callback test); a record without session / turn is
treated as a session of its own. Nothing here calls a model, and nothing
imports the rest of flow, so it runs wherever the data lives.

Definitions (the definitions are the point: change one and the numbers of
two files stop being comparable):

  language      of a reply: ja when it has kana, zh when it has Han and no
                kana, en when it has no CJK and at least 3 Latin words;
                otherwise None (counted in the totals, not in a language).
  words         English words of the spoken text: runs of Latin letters with
                inner ' or - ("don't", "well-known"). Counted for the English
                replies ("words/reply en").
  chars         per reply of ja and zh: characters of the spoken text without
                whitespace. "chars/reply (all)" is len(spoken) over every
                reply, spaces included (the spaces the marker gaps leave in
                a student's or teacher's spoken text count; a real
                streamer's file has none of them there): the number the
                first comparison of q8r3 with its teacher used.
  sentence      a stretch with a letter between the marks . ! ? their CJK
                forms, the ellipsis and the newline (flow.dataexport.
                sentence_count counts the same way). words/sentence for the
                English replies, chars/sentence for ja and zh.
  distinct-n    unique n-grams / n-grams over a whole file, none crossing a
                reply. distinct-1 / 2 (en): English words (lower-cased) of
                the English replies; distinct-2 (chars): single characters
                of the ja and zh replies. It falls as the file grows:
                compare files of about the same size (the 820-turn
                held-out set, or --sample N in scripts/style_ref.py).
  unique words  distinct English words (lower-cased) in the file.
  echo          the first sentence of the reply reuses at least 2 content
                words of the user line (English: not stopwords, 3 letters or
                more, a plural 's' dropped), or shares a substring of at
                least 4 characters with it (Japanese and Chinese). Over the
                replies whose user line is not empty and not the
                quiet-chat line.
  question      the spoken text ends with ? or a full-width ？ (a closing
                quote or bracket aside).
  opening       the first token of the spoken text: English up to the first
                space or punctuation mark (lower-cased); Japanese and
                Chinese the same, at most 3 characters. "top opening share"
                is the share of the most common one; the three most common
                are listed.
  interjection  the reply opens with an interjection of INTERJ (en: oh, ooh,
                hmm, wow, okay, ah, aw, hey, ha ...; ja えっ, あ, うん, わあ,
                ふふ ...; zh 嗯, 哇, 哎, 啊 ...).
  substance     the share of replies with at least one concrete item: a
                number (digits, a spelled-out number above one, a CJK numeral
                with a counter), a named thing (an English capitalised word
                inside a sentence, a katakana run of 3 or more, a quoted
                title in 「」『』《》“”), a first-person opinion or experience
                (en "I think / love / hate / used to / once / when I ...",
                ja 私は / 僕は ... 好き / 思う, 昔 / 前に, zh 我觉得 / 我喜欢 /
                我以前 ...), or a reference to an earlier turn of the session
                (a cue such as "earlier", "you said", さっき, 刚才, or at
                least 2 content words / a 4-character substring shared with
                an earlier user line and not with the current one). Each
                kind is reported on its own too. A proxy, not a judge: it
                counts a reply that names something, not one that says
                something true.
  length class  (beats3 rev 2) of a non-quiet reply, by its size (English
                words, ja / zh characters without whitespace; BANDS): short
                (up to 20 words / 35 chars), medium (up to 40 / 80), long
                (up to 60 / 130), over (more), each upper edge with 10%
                slack (SLACK). "length mix" is the share of each class;
                with a "target" on the records (the length the teacher was
                asked for, flow.dataset --length-mix) the hit rate is the
                share of replies whose class is their target, in total and
                per target.
  simile        a literary-register proxy, reported as a number and never as
                a check: simile markers per reply (en "like a / an",
                "as if", "as though", "as ... as a"; ja ような / みたいな /
                のように / かのよう; zh 像...一样 / 像...似的 / 的那种 / 仿佛 /
                宛如 / 如同 / 犹如) and the share of replies with at least one.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

QUIET_CHAT = "(chat is quiet)"          # flow.pose.QUIET_CHAT
_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_SENT = re.compile(r"[^.!?。！？…\n]*[^\s.!?。"
                   r"！？…][^.!?。！？…\n]*")
_CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")
_PUNCT_END = "\"'”’)」』）]】 \t\n"

STOP = frozenset("""
a an the and or but if so as of to in on at by for with from into about over
under than then that this these those there here is are was were be been
being am do does did done doing have has had having i me my mine myself you
your yours yourself he him his she her hers it its we us our they them their
what which who whom whose when where why how not no yes can could will would
shall should may might must just very really too also only even still much
many some any all each every more most other such own same here now out up
down off again once ever never get got gets getting go goes going went gone
let lets let's say said says tell told like want wants wanted need needs
thing things something anything everything one ok okay oh hey hi hello please
thanks thank im i'm you're don't it's that's can't
""".split())

# Interjection openings, matched at the start of the spoken text.
INTERJ = re.compile(
    r"^\W*(?:(?:oh|ooh|oooh|ohh|hmm+|hm+|mm+|wow|woah|whoa|okay|ok|ah+|aw+|"
    r"aww+|ha+|haha+|hehe+|eh|ehh|huh|hey|yay|oof|ugh|um+|uh+|well|wait|"
    r"yeah|yep|nope|aha|ahh|hee|whew|phew)\b|"
    r"えっ|えー|え、|え？|え！|あっ|あ、|あ！|あー|あぁ|あれ|うん|うーん|ううん|ん？|"
    r"んー|わあ|わぁ|わっ|わー|おお|おっ|ほう|へえ|へぇ|ふふ|あはは|えへへ|はは|"
    r"ええ|まあ|うわ|ふむ|ねえ|やった|"
    r"嗯|哇|哎|诶|欸|啊|哦|噢|哈哈|呀|唉|呃|咦|哎呀|喔|呜|嘿|哼|呵|嘻)", re.I)

_EN_NUMS = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|dozen|half)\b", re.I)
_NUM = re.compile(r"\d")
_CJK_NUM = re.compile(
    "[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e"
    "\u5343\u4e07\u4e24]+\\s*(?:个|個|点|點|年|岁|歲|次|天|人|杯|本|只|隻|块|塊|分|秒|"
    "小时|小時|回|つ|歳|日|冊|匹|枚|時)")
_KATAKANA = re.compile(r"[ァ-ヺー]{3,}")
_QUOTED = re.compile(r"[「『《“][^」』》”]"
                     r"{1,24}[」』》”]")
_OPINION_EN = re.compile(
    r"\bI(?:'ve| have|'d| would|'m| am)?\s+(?:really\s+|always\s+|"
    r"definitely\s+|kind of\s+|sort of\s+|honestly\s+|actually\s+)?"
    r"(?:think|thought|love|loved|like|liked|hate|hated|prefer|believe|"
    r"feel|felt|remember|remembered|used to|once|always|never|usually|"
    r"tried|went|made|got|had|was|grew|learned|learnt|started|stayed|"
    r"watched|played|read|wrote|cooked|baked|built|lost|found|saw|met|"
    r"miss|missed|wish|wished|bet|guess|swear|can't stand|enjoy|enjoyed|"
    r"adore|wonder|wondered|dream|dreamed)\b|"
    r"\b(?:when I|my (?:favou?rite|first|mom|mum|dad|grandma|grandpa|sister|"
    r"brother|friend|old|best|worst)|honestly,? (?:I|my)|in my (?:opinion|"
    r"book|view)|personally)\b", re.I)
_OPINION_JA = re.compile(
    r"(?:私|わたし|僕|ぼく|あたし|うち)[^。！？!?]{0,14}"
    r"(?:好き|嫌い|苦手|得意|思う|思います|思った|思ってる|思うよ|思うな|"
    r"覚えて|したことある|行ったこと|食べたこと|やったことある)|"
    r"(?:昔|子供の頃|子どもの頃|この前|前に)[^。！？!?]{2,}")
_OPINION_ZH = re.compile(
    r"我(?:觉得|覺得|认为|認為|喜欢|喜歡|爱|愛|讨厌|討厭|最|以前|曾|曾经|曾經|"
    r"小时候|小時候|记得|記得|想|猜|怕|常常|经常|經常|总是|總是|一直|上次|去年|"
    r"前几天|前幾天|有一次|吃过|吃過|去过|去過|试过|試過|看过|看過|玩过|玩過|"
    r"养过|養過|学过|學過)")
_CUE_EN = re.compile(
    r"\b(?:earlier|a (?:minute|moment|while) ago|last time|you (?:said|"
    r"mentioned|told me|asked)|as I said|like you said|remember (?:when|"
    r"that|what)|you were saying|going back to|back to what|the other "
    r"day|before you)\b", re.I)
_CUE_JA = re.compile(r"さっき|先ほど|前に言|言ってた|言ってい|言っていた|"
                     r"さっきの|あの時|あの話|覚えて|話してた|話してい|"
                     r"聞いた|前の話")
_CUE_ZH = re.compile(r"刚才|剛才|刚刚|剛剛|之前|你说过|你說過|你刚说|你剛說|"
                     r"还记得|還記得|上次|你提到|你说的|你說的|前面说|前面說|"
                     r"你问过|你問過")


# ---------------------------------------------------------------- lengths
# The size classes of a reply (flow.dataset's length plan asks the teacher for
# one per turn, flow.pose's rev 2 rules describe them): (words, characters)
# of the upper edge of each. English is measured in words, Japanese and
# Chinese in characters without whitespace.
BANDS = {"short": (20, 35), "medium": (40, 80), "long": (60, 130)}
LENGTH_TARGETS = tuple(BANDS)
SLACK = 1.1
LENGTH_CLASSES = LENGTH_TARGETS + ("over",)


def size_of(text):
    """(unit, n): ("chars", characters without whitespace) for a Japanese or
    Chinese reply, else ("words", English words)."""
    if lang_of(text) in ("ja", "zh"):
        return "chars", nchars(text)
    return "words", len(words(text))


def length_class(text) -> str:
    """short / medium / long / over for a reply (BANDS, SLACK)."""
    unit, n = size_of(text)
    k = 0 if unit == "words" else 1
    for name in LENGTH_TARGETS:
        if n <= BANDS[name][k] * SLACK:
            return name
    return "over"


def length_line(target, lang) -> str:
    """The teacher-only line of a turn's length plan, e.g. "(length for this
    reply: short -- one quick sentence, 20 words at most)" (an em dash); the
    unit follows the reply language `lang` (en words, ja / zh characters).
    It is added to the teacher's copy of the latest user message only and
    never to the record's user_content, tail or history, so the student
    never sees it (flow.dataset, docs/embodiment.md)."""
    cjk = lang in ("ja", "zh")
    lo, hi = {"short": ((0, 0), BANDS["short"]), "medium": (BANDS["short"],
              BANDS["medium"]), "long": (BANDS["medium"],
              BANDS["long"])}[target]
    k = 1 if cjk else 0
    unit = "characters" if cjk else "words"
    what = {"short": "one quick sentence, or two very short ones, "
                     f"{hi[k]} {unit} at most",
            "medium": f"two or three sentences, {lo[k]} to {hi[k]} {unit}, "
                      f"not fewer than {lo[k]}",
            "long": "tell it in full, a small story or a real explanation, "
                    f"at least {lo[k]} and up to {hi[k]} {unit}; do not stop "
                    "after one or two sentences"}[target]
    return f"(length for this reply: {target} \u2014 {what})"


LENGTH_MARK = "(length for this reply:"


_SIMILE = re.compile(
    r"\b(?:like\s+an?\b|as\s+if\b|as\s+though\b|as\s+\w+\s+as\s+(?:an?|the)\b)"
    r"|ような|のよう[なにで]|みたい[なにで]|かのよう|"
    r"像[^，。！？,.!?\n]{1,14}?(?:一样|一樣|似的|那样|那樣|般)|的那种|的那種|"
    r"仿佛|宛如|宛若|如同|犹如|猶如", re.I)


def similes(text) -> int:
    """Simile markers in a reply (module docstring): a number, not a check."""
    return len(_SIMILE.findall(text or ""))


def lang_of(text):
    """'ja', 'zh', 'en' or None (module docstring)."""
    kana = han = 0
    for ch in text or "":
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:
            kana += 1
        elif 0x3400 <= o <= 0x9FFF:
            han += 1
    if kana:
        return "ja"
    if han:
        return "zh"
    return "en" if len(_WORD.findall(text or "")) >= 3 else None


def words(text) -> list:
    return _WORD.findall(text or "")


def nchars(text) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def sentences(text) -> list:
    return [s.strip() for s in _SENT.findall(text or "") if s.strip()]


def tokens(text) -> list:
    """Lower-cased English words and single CJK characters in order."""
    out = []
    for m in re.finditer(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|" + _CJK.pattern,
                         text or ""):
        out.append(m.group(0).lower())
    return out


def content_words(text) -> set:
    out = set()
    for w in words(text):
        w = w.lower().replace("’", "'")
        if w in STOP or len(w) < 3:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def _cjk_overlap(a, b, n=4) -> bool:
    """Whether a and b share a substring of n CJK / kana characters."""
    a = "".join(_CJK.findall(a or ""))
    b = "".join(_CJK.findall(b or ""))
    if len(a) < n or len(b) < n:
        return False
    grams = {b[i:i + n] for i in range(len(b) - n + 1)}
    return any(a[i:i + n] in grams for i in range(len(a) - n + 1))


def echoes(first, user) -> bool:
    """The first sentence reuses the user line (module docstring)."""
    if not first or not (user or "").strip():
        return False
    if len(content_words(first) & content_words(user)) >= 2:
        return True
    return _cjk_overlap(first, user)


def opening(spoken):
    """The first token of a reply (module docstring), or None."""
    s = (spoken or "").lstrip(" \t\n\"'“‘「『([")
    if not s:
        return None
    m = _WORD.match(s)
    if m:
        return m.group(0).lower()
    m = re.match(r"[^\s,.!?;:、。，！？…~〜"
                 r"～\-—]+", s)
    return m.group(0)[:3] if m else None


def interjection(spoken) -> bool:
    return bool(INTERJ.match((spoken or "").lstrip()))


def ends_question(spoken) -> bool:
    return (spoken or "").rstrip(_PUNCT_END).endswith(("?", "？"))


def named_thing(text) -> bool:
    t = text or ""
    if _KATAKANA.search(t) or _QUOTED.search(t):
        return True
    # A capitalised word inside a sentence (not the first word, not "I").
    for s in sentences(t):
        ws = _WORD.findall(s)
        for w in ws[1:]:
            if w[0].isupper() and w not in ("I",) \
                    and not re.fullmatch(r"I['’](?:m|ve|d|ll)", w):
                return True
    return False


def has_number(text) -> bool:
    t = text or ""
    return bool(_NUM.search(t) or _EN_NUMS.search(t) or _CJK_NUM.search(t))


def has_opinion(text) -> bool:
    t = text or ""
    return bool(_OPINION_EN.search(t) or _OPINION_ZH.search(t)
                or _OPINION_JA.search(t))


def callback(spoken, user, earlier) -> bool:
    """The reply refers to an earlier turn: a cue, or words of an earlier
    user line that the current line does not have."""
    t = spoken or ""
    if _CUE_EN.search(t) or _CUE_JA.search(t) or _CUE_ZH.search(t):
        return True
    cur = content_words(user)
    mine = content_words(t)
    for e in earlier:
        if len((mine & content_words(e)) - cur) >= 2:
            return True
        if _cjk_overlap(t, e) and not _cjk_overlap(t, user):
            return True
    return False


def substance_kinds(spoken, user="", earlier=()) -> list:
    """The kinds of concrete item a reply has: number, named, opinion,
    callback."""
    out = []
    if has_number(spoken):
        out.append("number")
    if named_thing(spoken):
        out.append("named")
    if has_opinion(spoken):
        out.append("opinion")
    if callback(spoken, user, earlier):
        out.append("callback")
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def distinct(seqs, n) -> float | None:
    """unique n-grams / n-grams over token lists (none crosses a list)."""
    seen, total = set(), 0
    for s in seqs:
        for i in range(len(s) - n + 1):
            seen.add(tuple(s[i:i + n]))
            total += 1
    return len(seen) / total if total else None


def _earlier_lines(recs) -> dict:
    """{id(rec): [user lines of the session's earlier turns]}."""
    by = defaultdict(list)
    for r in recs:
        by[r.get("session") or id(r)].append(r)
    out = {}
    for rs in by.values():
        rs = sorted(rs, key=lambda r: r.get("turn") or 0)
        seen = []
        for r in rs:
            out[id(r)] = list(seen)
            u = (r.get("user") or "").strip()
            if u and u != QUIET_CHAT:
                seen.append(u)
    return out


def style(recs) -> dict:
    """The style numbers of these records (module docstring). Replies with
    nothing spoken are left out."""
    recs = [r for r in recs if (r.get("spoken") or "").strip()]
    n = len(recs)
    if not n:
        return {"n": 0}
    earlier = _earlier_lines(recs)
    by_lang = defaultdict(list)
    sent_n, wps, cps = [], [], defaultdict(list)
    seqs_en, seqs_cjk = [], []
    ends, ops, ints, echo_n, echo_d = 0, Counter(), 0, 0, 0
    sub, kinds = 0, Counter()
    uniq = set()
    sim, sim_any = 0, 0
    lens, hits, with_target = Counter(), Counter(), Counter()
    sub_by, sub_by_n = Counter(), Counter()
    for r in recs:
        sp = r["spoken"].strip()
        lg = lang_of(sp)
        by_lang[lg].append(r)
        ss = sentences(sp)
        sent_n.append(len(ss))
        w = words(sp)
        if lg == "en":
            wps.append(len(w) / max(1, len(ss)))
        elif lg in ("ja", "zh"):
            cps[lg].append(nchars(sp) / max(1, len(ss)))
        uniq.update(x.lower() for x in w)
        if lg == "en":
            seqs_en.append([x.lower() for x in w])
        elif lg in ("ja", "zh"):
            seqs_cjk.append(_CJK.findall(sp))
        ends += ends_question(sp)
        op = opening(sp)
        if op:
            ops[op] += 1
        ints += interjection(sp)
        u = (r.get("user") or "").strip()
        if u and u != QUIET_CHAT and ss:
            echo_d += 1
            echo_n += echoes(ss[0], u)
        ks = substance_kinds(sp, u, earlier[id(r)])
        sub += bool(ks)
        kinds.update(ks)
        k_ = similes(sp)
        sim += k_
        sim_any += bool(k_)
        if u != QUIET_CHAT:
            c_ = length_class(sp)
            lens[c_] += 1
            sub_by_n[c_] += 1
            sub_by[c_] += bool(ks)
            t_ = r.get("target")
            if t_ in BANDS:
                with_target[t_] += 1
                hits[t_] += c_ == t_
    top = ops.most_common(3)
    nops = sum(ops.values())
    out = {
        "n": n,
        "lang": {k or "other": len(v) for k, v in by_lang.items()},
        "words_en": _mean(len(words(r["spoken"])) for r in by_lang["en"]),
        "chars_ja": _mean(nchars(r["spoken"]) for r in by_lang["ja"]),
        "chars_zh": _mean(nchars(r["spoken"]) for r in by_lang["zh"]),
        "chars": _mean(len(r["spoken"]) for r in recs),
        "sentences": _mean(sent_n),
        "words_per_sentence_en": _mean(wps),
        "chars_per_sentence_ja": _mean(cps["ja"]),
        "chars_per_sentence_zh": _mean(cps["zh"]),
        "distinct1": distinct(seqs_en, 1),
        "distinct2": distinct(seqs_en, 2),
        "distinct2_chars": distinct(seqs_cjk, 2),
        "unique_words_en": len(uniq),
        "echo": echo_n / echo_d if echo_d else None,
        "question": ends / n,
        "top_opening_share": top[0][1] / nops if top else None,
        "top_openings": [(k, v / nops) for k, v in top],
        "interjection": ints / n,
        "substance": sub / n,
        "substance_kinds": {k: kinds[k] / n for k in
                            ("number", "named", "opinion", "callback")},
        "simile": sim / n,
        "simile_any": sim_any / n,
    }
    nl = sum(lens.values())
    out["length_mix"] = {k: lens[k] / nl for k in LENGTH_CLASSES} \
        if nl else None
    nt = sum(with_target.values())
    out["length_hit"] = sum(hits.values()) / nt if nt else None
    out["length_hit_by"] = {k: hits[k] / with_target[k] for k in BANDS
                            if with_target[k]}
    # A quick one-line reply is not meant to carry a story: substance read
    # by the length a reply came out (r4p2: 0.58 overall with 43% short).
    out["substance_by_length"] = {k: sub_by[k] / sub_by_n[k]
                                  for k in LENGTH_CLASSES if sub_by_n[k]}
    return out


def _sub_by(st):
    m = st.get("substance_by_length") or {}
    if not m:
        return "-"
    return " / ".join(f"{m[k]:.2f}" if k in m else "-"
                      for k in ("short", "medium", "long"))


def _mix(m):
    return " / ".join(f"{m[k]:.0%}" for k in LENGTH_CLASSES) if m else None


def _hit(st):
    """The hit rate of the length plan ("62% (short 70%, medium 55%, long
    48%)"), None for records without targets."""
    if st.get("length_hit") is None:
        return None
    by = ", ".join(f"{k} {v:.0%}" for k, v in
                   (st.get("length_hit_by") or {}).items())
    return f"{st['length_hit']:.0%} ({by})"


def rows_of(st) -> list:
    """[(label, value)] for the table: floats, or a string."""
    if not st or not st.get("n"):
        return []
    tops = ", ".join(f"{k} {v:.0%}" for k, v in st["top_openings"])
    sk = st["substance_kinds"]
    return [
        ("style: replies", st["n"]),
        ("  words/reply (en)", st["words_en"]),
        ("  chars/reply (ja)", st["chars_ja"]),
        ("  chars/reply (zh)", st["chars_zh"]),
        ("  chars/reply (all, len)", st["chars"]),
        ("  sentences/reply", st["sentences"]),
        ("  words/sentence (en)", st["words_per_sentence_en"]),
        ("  chars/sentence (ja)", st["chars_per_sentence_ja"]),
        ("  chars/sentence (zh)", st["chars_per_sentence_zh"]),
        ("  distinct-1 (en)", st["distinct1"]),
        ("  distinct-2 (en)", st["distinct2"]),
        ("  distinct-2 (ja+zh chars)", st["distinct2_chars"]),
        ("  unique words (en)", st["unique_words_en"]),
        ("  echo of the user line", st["echo"]),
        ("  ends with a question", st["question"]),
        ("  top opening share", st["top_opening_share"]),
        ("  top openings", tops),
        ("  interjection opening", st["interjection"]),
        ("  substance (any)", st["substance"]),
        ("    number", sk["number"]),
        ("    named thing", sk["named"]),
        ("    opinion / experience", sk["opinion"]),
        ("    callback", sk["callback"]),
        ("  substance by length (s/m/l)", _sub_by(st)),
        ("  length short/medium/long/over", _mix(st.get("length_mix"))),
        ("  length target hit rate", _hit(st)),
        ("  similes/reply (literary proxy)", st.get("simile")),
        ("  replies with a simile", st.get("simile_any")),
    ]
