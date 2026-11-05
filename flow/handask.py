"""
Whether a user line explicitly asks the character for a hand or arm action:
point, wave, raise or hold up a hand, a peace sign, a thumbs up, clap, hands
on the hips, cross the arms, cover the face, a heart, a size shown with the
hands, a salute (en / ja / zh, simplified and traditional).

One detector for flow.checks (body_request_obeyed, wave_obeyed) and
flow.dataexport (the skipped_request flag). It is stricter than
dataexport.asked_for, which decides what the manners keep and so errs toward
keeping a hand: this one asks for an instruction, not a mention.

  - the line must say the action as a request: an imperative or a question
    to the character ("point at the tree", "can you wave", "手を振って",
    "给我比个耶"). The topic is no request: an invitation, a tease or small
    talk about something else asks for no hand. A mention is none either:
    in the past, as a compliment, as a noun ("your peace sign was cute",
    "that peace sign photo is a keeper", "指差してる姿", "你比耶的样子");
  - a request for the head or the eyes alone (look, turn, nod, follow with
    your eyes) is none; no pattern here matches one;
  - the topic only helps for show / pose, whose lines are requests by
    design: there a bare hand, arm or finger word counts as the action word.

Heuristic, extended by adding patterns to the lists below.
"""

from __future__ import annotations

import re
import unicodedata

SHOW_TOPICS = ("show", "pose")

# What stands before an English verb that asks: the start of the line or a
# clause, a polite frame, or a conjunction ("... and point to the left").
_CTX = (r"(?:^|[.!?,;:\-—…(\"'\n]\s*|\b(?:please|pls|plz|can you|"
        r"could you|would you|will you|won't you|why don't you|why not|"
        r"how about you|how about|let me see you|let's see you|see you|"
        r"watch you|i want you to|i'd like you to|i'd love you to|"
        r"i'd love to see you|i wanna see you|go ahead and|now|then|and|"
        r"just|also|try to|try|first|quickly|again|maybe)\s+)")
_OBJ = r"(?:your|a|an|one|that|both|the|those|ur|my)?\s*(?:\w+\s+){0,2}?"
_LIMB = r"(?:hands?|arms?|fingers?|fists?|paws?)\b"
_SIGN = (r"(?:peace\s+sign|v[- ]sign|victory\s+sign|thumbs?[- ]?up|"
         r"high[- ]?(?:five|fives)|give\s+me\s+five|salute)")
EN_ASK = [
    # point at / to / toward ...; pointing the eyes or the head is no hand
    _CTX + r"point(?:ing)?\s+(?:to|at|toward|towards|over|me|us|it|where|"
    r"that|this|the|something|someone|up|down|left|right|out\s+(?:the|where|"
    r"which|what)|(?:your|a|one|that|the)\s+(?:finger|hand|arm|index))\b"
    r"(?!\s+(?:eyes|head|gaze|face))",
    _CTX + r"wav(?:e|ing)\b",
    _CTX + r"(?:clap|clapping|salute|saluting|snap)\b",
    # raise / hold up / lift / put up / stick out a limb
    _CTX + r"(?:rais(?:e|ing)|lift(?:ing)?|hold(?:ing)?\s+up|put(?:ting)?"
    r"\s+up|stick(?:ing)?\s+out|stretch(?:ing)?\s+out|keep\s+(?:that|your|"
    r"the))\s+" + _OBJ + _LIMB,
    r"\bhands?\s+up\b(?!\s+(?:if|who|when))",
    # hands on hips, crossed arms, hands together, a covered face, open arms
    r"\bhands?\s+on\s+(?:your\s+|ur\s+)?(?:hips?|waist)\b",
    r"\b(?:put|place)\s+(?:your\s+|both\s+)?hands?\s+on\b",
    _CTX + r"(?:cross|fold)(?:ing)?\s+(?:your\s+|both\s+|those\s+)?"
    r"(?:arms?|hands?|fingers?)\b",
    r"\barms?\s+(?:crossed|folded)\b",
    r"\bhands?\s+together\b",
    _CTX + r"(?:cover|hide|covering|hiding)\s+(?:your\s+)?(?:face|mouth|"
    r"eyes)\b",
    _CTX + r"(?:open|spread|opening|spreading)\s+(?:your\s+|both\s+)?arms\b",
    # a sign named as a thing to give, show or make
    r"\b(?:give|show|make|do|flash|throw|strike|pull|put\s+up|hold\s+up|"
    r"let\s+me\s+see|how\s+about|can\s+i\s+get|c'mon)\b[^.!?\n]{0,25}?\b"
    + _SIGN + r"\b",
    r"\b" + _SIGN + r"\s*(?:please|pls|plz|for\s+(?:me|the\s+camera|us|chat)"
    r"|to\s+the\s+camera)",
    # a heart made with the hands
    r"\b(?:make|do|form|draw|shape)\b[^.!?\n]{0,12}?\bheart\b",
    r"\bheart\b\s*(?:with|using)\s+(?:your\s+)?" + _LIMB,
    r"\b(?:finger|hand|arm)\s+heart\b",
    # with your hands: a size, a shape; a bow with the hands
    r"\b(?:with|using)\s+(?:both\s+|your\s+|ur\s+|those\s+)+" + _LIMB,
    r"\b(?:bow|curtsy)\b[^.!?\n]{0,20}?\bhands?\b",
    r"\bshake\s+(?:my|your)\s+hand\b",
    r"\btouch\s+(?:your|ur)\s+(?:nose|cheek|face|hair|ear|chin|head|"
    r"forehead|lips?)\b",
]

# A gesture word of Japanese, then a request ending, and not the progressive
# or the past ("指差してる姿", "手を振ってた").
_JA_STEM = (r"(?:指差|指さ|ゆびさ|指を差|指をさ|指で指|指で差|手を振|手をふ|"
            r"手を上げ|手を挙げ|手をあげ|腕を上げ|腕を挙げ|腕を組|腕組|手を叩|"
            r"手をたた|手を合わせ|手を腰|腰に手|手を伸ばし|手を広げ|両手を|"
            r"片手を|手で隠|顔を隠|顔を覆|ピース|サムズアップ|ハイタッチ|"
            r"ハートを?|敬礼|拍手|万歳|バンザイ|ばんざい|手を出|手を差し出|"
            r"手のひらを|指を立て)")
_JA_REST = (r"[をはにでもっ]{0,2}(?:て(?:くだ|くれ|ほし|よ|ね|み|ちょ|$|"
            r"[!！。、\s?？])|って(?:くだ|くれ|ほし|よ|ね|み|ちょ|$|"
            r"[!！。、\s?？])|して(?!る|い|た)|し?てみ|しよう|しなよ|ろう|ください|"
            r"下さい|ほしい|欲しい|ちょうだい|見せ|みせ|作って|つくって|"
            r"やって|お願い|おねがい|出して)")
JA_ASK = [_JA_STEM + _JA_REST]

# Chinese (simplified and traditional): a gesture word not followed by 的
# (a modifier: "比耶的样子"), 过 / 了 (past) or a noun of a picture.
_ZH_GESTURE = (r"(?:指向|指着|指著|指一下|指一指|指指|指给|指給|指那|指这|"
               r"指這|用手指|手指着|手指著|手指向|[挥揮]\s*[挥揮]?\s*手|"
               r"招\s*招?\s*手|[举舉]\s*[举舉]?\s*手|"
               r"[举舉]起(?:你的|右|左|双|雙)?[手胳臂]|[举舉]高(?:你的)?手|"
               r"[举舉][个個]手|抬起?[手胳]|伸出?[手胳]|比[个個]?耶|"
               r"比[个個]?[Vv]|剪刀手|比[个個]?心|比[个個]?[爱愛]心|"
               r"比[赞讚]|[竖豎]起?大拇指|[点點][赞讚]|鼓掌|拍手|拍拍手|"
               r"叉腰|抱胸|抱臂|交叉.?[双雙]?[手臂]|[双雙]臂交叉|"
               r"[捂遮]住?[脸臉嘴]|[双雙]手合十|敬[礼禮]|比[划劃画畫]|"
               r"[摆擺][个個]?[手姿]|比[个個]?手[势勢])")
ZH_ASK = [_ZH_GESTURE + r"(?!的|过|過|了|时|時|姿|样|樣|照|图|圖|片)"]

# Words with a limb character inside that name no limb (the compounds of
# dataexport.ASK_NOT, the part the weak rule needs).
_NOT = [
    r"on the other hand", r"second[- ]hand", r"hands? down", r"by hand",
    r"upper hand", r"out of hand", r"\bin hand\b", r"hands[- ]on",
    r"相手", r"上手", r"下手", r"助手", r"苦手", r"歌手", r"選手", r"手伝",
    r"手紙", r"手作り", r"手間", r"手術", r"手に入", r"手前", r"手軽", r"得手",
    r"運転手", r"入手", r"投手", r"打手", r"騎手", r"手帳", r"手段", r"手続",
    r"目指", r"指導", r"指示", r"指摘", r"指定", r"指名", r"指針", r"指令",
    r"指揮", r"指標", r"腕前", r"腕時計", r"手机", r"对手", r"高手", r"手艺",
    r"手续", r"手术", r"手表", r"手套", r"选手", r"新手", r"手头", r"举例",
    r"举办", r"举行", r"举报", r"举止", r"指导", r"指示", r"指出", r"指定",
    r"指望", r"指标", r"指南", r"指责", r"指挥", r"手機", r"對手", r"指導",
    r"舉例", r"舉辦", r"舉行", r"舉止"]
# show / pose topics: a line that is a request by design needs only a word
# that names a limb.
_WEAK = [r"\b(?:hands?|arms?|fingers?)\b", r"[手腕指]", r"[挥揮举舉]"]

_ASK = [re.compile(p, re.I) for p in EN_ASK + JA_ASK + ZH_ASK]
_CUT = [re.compile(p, re.I) for p in _NOT]
_WEAKRE = [re.compile(p, re.I) for p in _WEAK]
_WAVE_ZH = re.compile(r"(?:[挥揮]\s*[挥揮]?\s*手|招\s*招?\s*手)"
                      r"(?!的|过|過|了|时|時|姿|样|樣)")
_WAVE_JA = re.compile(r"(?:手を振|手をふ)" + _JA_REST)
_WAVE_EN = re.compile(_CTX + r"wav(?:e|ing)\b", re.I)


def _norm(line) -> str:
    return unicodedata.normalize("NFKC", line or "")


def asked_hand_action(line, topic=None) -> bool:
    """True when the line explicitly asks for a hand or arm action (module
    docstring). topic: for show / pose a bare limb word is enough too."""
    s = _norm(line)
    if any(p.search(s) for p in _ASK):
        return True
    if topic in SHOW_TOPICS:
        for p in _CUT:
            s = p.sub(" ", s)
        return any(p.search(s) for p in _WEAKRE)
    return False


def asked_wave_action(line) -> bool:
    """True when the line asks the character to wave (en wave, ja 手を振って
    / バイバイ, zh 挥手 / 揮手), not a mention of a wave."""
    s = _norm(line)
    return bool(_WAVE_EN.search(s) or _WAVE_JA.search(s)
                or re.search(r"バイバイ|ばいばい", s) or _WAVE_ZH.search(s))
