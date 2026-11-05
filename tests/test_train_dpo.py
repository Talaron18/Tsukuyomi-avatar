"""
flow.train.dpo. The data layer, the loss and the plan of a run need no torch:
a character tokenizer stands in for the model's (the masking, the cut of the
prompt, the split, the data hash and the refusal to resume, the CLI's stats).
The model-side tests (the reference with the adapter off, the cached
reference, steps that raise the margin, a whole run, merge) run on a tiny
random Qwen3 (tests/tiny_lm.py) on CPU and skip without torch, transformers,
peft and tokenizers. The 4-bit path needs a GPU and is not run here.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from flow.train import chat
from flow.train import dpo
from flow.train import sft

SYSTEM = "You are Hiyori. Write [pose {...}] markers."


class CharTok:
    """One token a character; what sft.encode needs of a tokenizer."""
    chat_template = None

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in s]}


def prompt_of(i, turns=2):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in range(turns):
        msgs.append({"role": "user", "content": f"hello {i} {t}"})
        msgs.append({"role": "assistant", "content": f"old reply {i} {t}"})
    msgs.append({"role": "user", "content": f"now {i}"})
    return msgs


def write_pairs(path, n=20, kinds=("look_flip", "wave_frame"), same=0,
                sessions=5):
    rows, metas = [], []
    for i in range(n):
        rows.append({"prompt": prompt_of(i),
                     "chosen": [{"role": "assistant", "content": f"good {i}"}],
                     "rejected": [{"role": "assistant",
                                   "content": f"bad {i}" if i >= same
                                   else f"good {i}"}]})
        metas.append({"id": f"S{i % sessions}/t{i:02d}",
                      "avatar": "Hiyori" if i % 2 else "Mao",
                      "kind": f"synthetic:{kinds[i % len(kinds)]}"})
    d = os.path.dirname(str(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    with open(dpo.meta_path(str(path)), "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(m) + "\n" for m in metas))
    return rows, metas


# ---------------------------------------------------------------- the loss

def test_loss_math_on_hand_made_logps():
    # equal to the reference: log 2, accuracy a tie
    loss, st = dpo.dpo_loss(-10.0, -20.0, -10.0, -20.0, 5, 5, beta=0.1)
    assert loss == pytest.approx(math.log(2))
    assert st["margin"] == 0 and st["accuracy"] == 0.5
    # the policy prefers chosen by 4 nats more than the reference: margin
    # beta * 4, loss -log sigmoid(beta * 4), accuracy 1
    loss, st = dpo.dpo_loss(-8.0, -22.0, -10.0, -20.0, 5, 5, beta=0.1)
    assert st["rewards/chosen"] == pytest.approx(0.2)
    assert st["rewards/rejected"] == pytest.approx(-0.2)
    assert st["margin"] == pytest.approx(0.4) and st["accuracy"] == 1.0
    assert loss == pytest.approx(-math.log(1 / (1 + math.exp(-0.4))))
    # the other way round: bigger loss, accuracy 0
    worse, st2 = dpo.dpo_loss(-12.0, -18.0, -10.0, -20.0, 5, 5, beta=0.1)
    assert worse > math.log(2) > loss and st2["accuracy"] == 0.0
    # beta scales the margin
    _l, st3 = dpo.dpo_loss(-8.0, -22.0, -10.0, -20.0, 5, 5, beta=0.5)
    assert st3["margin"] == pytest.approx(2.0)
    # the sign: only the difference of the log-ratios counts, a drop of
    # both replies' log-probs by the same amount changes nothing
    l1, _ = dpo.dpo_loss(-8.0, -22.0, -10.0, -20.0)
    l2, _ = dpo.dpo_loss(-18.0, -32.0, -20.0, -30.0)
    assert l1 == pytest.approx(l2)


def test_sft_alpha_adds_the_chosen_mean_nll_and_length_norm_divides():
    base, _ = dpo.dpo_loss(-8.0, -22.0, -10.0, -20.0, 4, 10, beta=0.1)
    with_sft, st = dpo.dpo_loss(-8.0, -22.0, -10.0, -20.0, 4, 10, beta=0.1,
                                sft_alpha=0.5)
    assert st["nll"] == pytest.approx(2.0)          # 8 nats over 4 tokens
    assert with_sft == pytest.approx(base + 0.5 * 2.0)
    # per token: (2/4 - -2/10) * beta
    _l, st = dpo.dpo_loss(-8.0, -22.0, -10.0, -20.0, 4, 10, beta=0.1,
                          length_norm=True)
    assert st["margin"] == pytest.approx(0.1 * (2 / 4 + 2 / 10))
    # a numerically large margin does not overflow
    big, _ = dpo.dpo_loss(0.0, -1000.0, -1000.0, 0.0, 1, 1, beta=1.0)
    assert big == pytest.approx(0.0, abs=1e-9)
    huge, _ = dpo.dpo_loss(-1000.0, 0.0, 0.0, -1000.0, 1, 1, beta=1.0)
    assert huge == pytest.approx(2000.0)


def test_loss_on_tensors_matches_the_floats():
    torch = pytest.importorskip("torch")
    t = lambda x: torch.tensor(x, dtype=torch.float32)       # noqa: E731
    a, sa = dpo.dpo_loss(t(-8.0), t(-22.0), t(-10.0), t(-20.0), 4, 10, 0.1,
                         0.3)
    b, sb = dpo.dpo_loss(-8.0, -22.0, -10.0, -20.0, 4, 10, 0.1, 0.3)
    assert float(a) == pytest.approx(b, rel=1e-6)
    assert sa["margin"] == pytest.approx(sb["margin"], rel=1e-6)


# -------------------------------------------------------------------- data

def test_specs_and_weights(tmp_path):
    assert dpo.parse_specs("a.jsonl") == [("a.jsonl", 1.0)]
    assert dpo.parse_specs("a.jsonl:1,runs/b.jsonl:2.5") == [
        ("a.jsonl", 1.0), ("runs/b.jsonl", 2.5)]
    assert dpo.parse_specs("C:\\x\\a.jsonl") == [("C:\\x\\a.jsonl", 1.0)]
    with pytest.raises(SystemExit):
        dpo.parse_specs("")
    write_pairs(tmp_path / "dpo.jsonl", 4)
    pairs, _ = dpo.load_pairs(str(tmp_path), 2.0)             # a folder
    assert len(pairs) == 4 and len(dpo.expand(pairs)) == 8
    half = dpo.expand([dict(p, weight=0.5) for p in pairs * 50], seed=3)
    assert 60 < len(half) < 140                               # a chance
    assert dpo.expand([dict(p, weight=0) for p in pairs]) == []


def test_pairs_with_meta_and_identical_ones_dropped(tmp_path):
    rows, _ = write_pairs(tmp_path / "dpo.jsonl", 10, same=3)
    pairs, info = dpo.load_pairs(str(tmp_path / "dpo.jsonl"))
    assert info["same_text"] == 3 and len(pairs) == 7
    p = pairs[0]
    assert p["kind"].startswith("synthetic:") and p["avatar"] in (
        "Hiyori", "Mao")
    assert p["session"] == p["id"].rsplit("/", 1)[0]
    assert p["chosen"] == "good 3" and p["rejected"] == "bad 3"
    assert p["prompt"] == rows[3]["prompt"]
    # no meta: kinds unknown, a session from the first user line
    os.remove(dpo.meta_path(str(tmp_path / "dpo.jsonl")))
    pairs, info = dpo.load_pairs(str(tmp_path / "dpo.jsonl"))
    assert info["no_meta"] == 10 and {q["kind"] for q in pairs} == {"unknown"}
    assert len({q["session"] for q in pairs}) > 1


def test_split_by_session_and_the_sft_eval_sessions_stay_out(tmp_path):
    write_pairs(tmp_path / "dpo.jsonl", 60, sessions=12)
    pairs, _ = dpo.load_pairs(str(tmp_path / "dpo.jsonl"))
    tr, ev, info = dpo.split_pairs(pairs, 0.25)
    assert not {p["session"] for p in tr} & {p["session"] for p in ev}
    assert ev and len(tr) + len(ev) == 60
    # the SFT run's held-out sessions are in neither side
    out = {"S0", "S1"}
    tr2, ev2, info2 = dpo.split_pairs(pairs, 0.25, sft_eval=out)
    assert not {p["session"] for p in tr2 + ev2} & out
    assert info2["sft_eval_dropped"] == 10
    assert dpo.split_pairs(pairs, 0.25, sft_eval={f"S{i}" for i in range(12)}
                           )[:2] == ([], [])


# ------------------------------------------------------------------ tokens

def test_masking_is_sfts_for_the_same_messages():
    tok = CharTok()
    pair = {"prompt": prompt_of(1), "chosen": "good 1", "rejected": "bad 1",
            "kind": "k", "avatar": "a"}
    row, cut = dpo.encode_pair(tok, pair, "qwen3")
    assert cut == 0
    # SFT's per-turn example of the same turn: the history is weight 0
    msgs = [dict(m, weight=0) if m["role"] == "assistant" else dict(m)
            for m in pair["prompt"]] + [{"role": "assistant",
                                         "content": "good 1"}]
    ref = sft.encode(tok, sft.clean({"messages": msgs}), "qwen3")
    assert row["chosen_ids"] == ref["input_ids"]
    assert row["chosen_labels"] == ref["labels"]
    # the full text is the chat module's, the empty think block included
    text, spans = chat.render(msgs, "qwen3")
    assert "<think>\n\n</think>\n\ngood 1<|im_end|>" in text
    assert "".join(chr(i) for i in row["chosen_ids"]) == text
    # exactly one labelled run: the reply and its end token
    lab = [i for i, x in zip(row["chosen_ids"], row["chosen_labels"])
           if x != -100]
    assert "".join(chr(i) for i in lab) == "good 1<|im_end|>"
    lab = [i for i, x in zip(row["rejected_ids"], row["rejected_labels"])
           if x != -100]
    assert "".join(chr(i) for i in lab) == "bad 1<|im_end|>"
    # the two sides share the prompt, token for token
    n = next(i for i, x in enumerate(row["chosen_labels"]) if x != -100)
    assert row["chosen_ids"][:n] == row["rejected_ids"][:n]
    assert dpo.n_labelled(row["chosen_labels"]) == len("good 1<|im_end|>")


def test_truncation_cuts_old_history_never_the_replies_or_the_system():
    tok = CharTok()
    pair = {"prompt": prompt_of(1, turns=6), "chosen": "good 1 " + "x" * 30,
            "rejected": "bad 1", "kind": "k", "avatar": "a"}
    whole, _ = dpo.encode_pair(tok, pair, "qwen3")
    n = len(whole["chosen_ids"])
    row, cut = dpo.encode_pair(tok, pair, "qwen3", max_len=n - 60)
    assert cut > 0 and len(row["chosen_ids"]) <= n - 60
    text = "".join(chr(i) for i in row["chosen_ids"])
    assert SYSTEM in text and "now 1" in text            # kept
    assert "hello 1 0" not in text and "hello 1 5" in text   # oldest first
    assert text.endswith("good 1 " + "x" * 30 + "<|im_end|>\n")
    lab = "".join(chr(i) for i, x in zip(row["chosen_ids"],
                                         row["chosen_labels"]) if x != -100)
    assert lab == "good 1 " + "x" * 30 + "<|im_end|>"
    # both sides keep the same prompt
    k = next(i for i, x in enumerate(row["chosen_labels"]) if x != -100)
    assert row["chosen_ids"][:k] == row["rejected_ids"][:k]
    # not even the system + last user + reply fit: the pair is left out
    gone, c = dpo.encode_pair(tok, pair, "qwen3", max_len=40)
    assert gone is None and c > 0
    rows, st = dpo.encode_all(tok, [pair, pair], "qwen3", n - 60)
    assert len(rows) == 2 and st["cut_pairs"] == 1 and st["cut_turns"] >= 1
    rows, st = dpo.encode_all(tok, [pair], "qwen3", 40)
    assert rows == [] and st["dropped"] == 1


def test_length_stats_count_what_would_be_cut():
    tok = CharTok()
    pairs = [{"prompt": prompt_of(i, turns=1 + 3 * (i % 2)),
              "chosen": "good", "rejected": "bad", "kind": "k",
              "avatar": "a"} for i in range(4)]
    rep = dpo.length_stats(tok, pairs, "qwen3", max_len=400)
    assert rep["pairs"] == 4 and rep["chosen"]["n"] == 4
    assert rep["over_max_len"] == 2 and rep["cannot_fit"] == 0
    rep = dpo.length_stats(tok, pairs, "qwen3", max_len=20)
    assert rep["cannot_fit"] == 4
    assert "cannot fit" in dpo.length_text(rep)


# -------------------------------------------------- the plan, hash, resume

def fake_sft_run(path, eval_sessions=("S0",), data="data/r3"):
    os.makedirs(path, exist_ok=True)
    for name, body in (
            ("split", {"eval_sessions": list(eval_sessions),
                       "train_sessions": ["S1", "S2", "S3", "S4"]}),
            ("data", {"data": data, "hash": "x"}),
            ("train", {"family": "qwen3", "base": "Qwen/Qwen3-8B"})):
        with open(os.path.join(path, f"{name}.json"), "w") as fh:
            json.dump(body, fh)
    return str(path)


def argv_of(tmp_path, *extra):
    return ["train", "--sft-run", str(tmp_path / "sft"), "--data",
            str(tmp_path / "pairs" / "dpo.jsonl"), "--out",
            str(tmp_path / "run"), "--base", "unused", "--max-len", "4096",
            "--eval-frac", "0.3", *extra]


def test_plan_writes_the_runs_files_and_refuses_other_data(tmp_path):
    fake_sft_run(tmp_path / "sft")
    write_pairs(tmp_path / "pairs" / "dpo.jsonl", 30, sessions=5)
    o = dpo.resolve(dpo.parse(argv_of(tmp_path)))
    pl = dpo.plan(o, CharTok())
    out = tmp_path / "run"
    # the eval script's inputs: the SFT run's split, the data folder
    assert json.load(open(out / "split.json")) == json.load(
        open(tmp_path / "sft" / "split.json"))
    d = json.load(open(out / "data.json"))
    assert d["data"] == "data/r3" and d["hash"] == pl["hash"]
    assert d["train_rows"] == len(pl["rows_tr"]) > 0 and pl["rows_ev"]
    # pairs of the SFT run's held-out session S0 are not trained on
    assert d["train_rows"] + d["eval_rows"] == 24
    assert json.load(open(out / "dpo_split.json"))["sft_eval_dropped"] == 6
    again = dpo.plan(dpo.resolve(dpo.parse(argv_of(tmp_path))), CharTok())
    assert again["hash"] == pl["hash"]                      # deterministic
    # a checkpoint of this data resumes; of other data it is refused
    os.makedirs(out / "checkpoints" / "checkpoint-5")
    assert dpo.plan(dpo.resolve(dpo.parse(argv_of(tmp_path))),
                    CharTok())["ck"].endswith("checkpoint-5")
    write_pairs(tmp_path / "pairs" / "dpo.jsonl", 32, sessions=5)
    with pytest.raises(SystemExit, match="other data"):
        dpo.plan(dpo.resolve(dpo.parse(argv_of(tmp_path))), CharTok())
    # --restart drops the checkpoints
    o = dpo.resolve(dpo.parse(argv_of(tmp_path, "--restart")))
    assert dpo.plan(o, CharTok())["ck"] is None
    assert not (out / "checkpoints").exists()


def test_weights_change_the_hash_and_all_eval_sessions_say_how_to_fix(tmp_path):
    fake_sft_run(tmp_path / "sft", eval_sessions=("S0", "S1", "S2", "S3",
                                                  "S4"))
    write_pairs(tmp_path / "pairs" / "dpo.jsonl", 10)
    with pytest.raises(SystemExit, match="--side train"):
        dpo.plan(dpo.resolve(dpo.parse(argv_of(tmp_path))), CharTok())
    o = dpo.resolve(dpo.parse(argv_of(tmp_path, "--allow-eval-sessions")))
    h1 = dpo.plan(o, CharTok())["hash"]
    a = argv_of(tmp_path, "--allow-eval-sessions")
    a[a.index("--data") + 1] += ":2"
    h2 = dpo.plan(dpo.resolve(dpo.parse(a)), CharTok())["hash"]
    assert h1 != h2


def test_stats_cli(tmp_path, monkeypatch, capsys):
    fake_sft_run(tmp_path / "sft")
    write_pairs(tmp_path / "pairs" / "dpo.jsonl", 30, sessions=5)
    monkeypatch.setattr(sft, "load_tokenizer", lambda name: CharTok())
    js = tmp_path / "stats.json"
    assert dpo.main(["stats", "--sft-run", str(tmp_path / "sft"), "--data",
                     str(tmp_path / "pairs" / "dpo.jsonl"), "--max-len",
                     "200", "--json", str(js)]) == 0
    out = capsys.readouterr().out
    assert "synthetic:look_flip" in out and "synthetic:wave_frame" in out
    assert "Hiyori" in out and "Mao" in out
    assert "held-out sessions" in out and "pairs over" in out
    rep = json.load(open(js))["rep"]
    assert rep["pairs"] == 24 and rep["over_max_len"] > 0


def test_merge_gguf_modelfile_are_sfts(monkeypatch):
    seen = []
    monkeypatch.setattr(sft, "main", lambda argv: seen.append(argv) or 0)
    assert dpo.main(["merge", "--run", "r"]) == 0
    assert dpo.main(["modelfile", "--run", "r", "--name", "tsk-q8r3d"]) == 0
    assert seen == [["merge", "--run", "r"],
                    ["modelfile", "--run", "r", "--name", "tsk-q8r3d"]]


# --------------------------------------------------- the model (tiny, CPU)

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("peft")
    from tests import tiny_lm
    d = tmp_path_factory.mktemp("lm") / "tiny-qwen3"
    texts = [chat.render(prompt_of(i) + [{"role": "assistant",
                                          "content": f"good {i}"}],
                         "qwen3")[0] for i in range(20)]
    return tiny_lm.build(str(d), texts * 3)


def peft_model(tiny, seed=0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(tiny)
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=sft.TARGETS,
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def rows_for(tiny, n=3):
    tok = sft.load_tokenizer(tiny)
    pairs = [{"prompt": prompt_of(i), "chosen": f"good {i}",
              "rejected": f"bad {i}", "kind": "k", "avatar": "a"}
             for i in range(n)]
    return dpo.encode_all(tok, pairs, "qwen3", None)[0]


def test_tokenization_equals_sft_with_the_real_tokenizer(tiny):
    tok = sft.load_tokenizer(tiny)
    pair = {"prompt": prompt_of(2), "chosen": "good 2", "rejected": "bad 2",
            "kind": "k", "avatar": "a"}
    row, _ = dpo.encode_pair(tok, pair, "qwen3")
    msgs = [dict(m, weight=0) if m["role"] == "assistant" else dict(m)
            for m in pair["prompt"]] + [{"role": "assistant",
                                         "content": "good 2"}]
    ref = sft.encode(tok, sft.clean({"messages": msgs}), "qwen3")
    assert row["chosen_ids"] == ref["input_ids"]
    assert row["chosen_labels"] == ref["labels"]
    assert sum(x != -100 for x in row["chosen_labels"]) == dpo.n_labelled(
        row["chosen_labels"])


def test_reference_with_the_adapter_off_is_the_base_model(tiny):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(tiny)
    model = peft_model(tiny)
    # a trained adapter: B is not zero any more
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.5)
    row = rows_for(tiny, 1)[0]
    t = dpo._tensors(row, "cpu")
    with torch.no_grad():
        want, _ = dpo.seq_logp(base, *t["chosen"])
        on, _ = dpo.seq_logp(model, *t["chosen"])
        with model.disable_adapter():
            off, _ = dpo.seq_logp(model, *t["chosen"])
    assert float(off) == pytest.approx(float(want), abs=1e-4)
    assert abs(float(on) - float(want)) > 1e-3
    # logits_to_keep and the full head agree
    with torch.no_grad():
        full, n1 = dpo.seq_logp(base, *t["chosen"], keep_logits=False)
        part, n2 = dpo.seq_logp(base, *t["chosen"], keep_logits=True)
    assert n1 == n2 and float(full) == pytest.approx(float(part), abs=1e-4)


def test_precomputed_reference_is_reused(tiny, tmp_path, monkeypatch):
    model = peft_model(tiny)
    rows = rows_for(tiny, 3)
    path = str(tmp_path / "ref_logps.json")
    meta = {"version": 1, "base": "b", "family": "qwen3", "max_len": 1}
    ref = dpo.reference_logps(model, rows, path, meta, logf=lambda s: None)
    assert set(ref) == {r["key"] for r in rows}
    # a fresh LoRA is the reference: the policy has the same log-probs
    c, r = dpo.policy_logps(model, rows[0], "cpu")
    assert float(c) == pytest.approx(ref[rows[0]["key"]][0], abs=1e-4)
    assert dpo.check_start(model, rows[0], ref, logf=lambda s: None) < 1e-3

    def boom(*a, **k):
        raise AssertionError("recomputed")
    monkeypatch.setattr(dpo, "policy_logps", boom)
    again = dpo.reference_logps(model, rows, path, meta, logf=lambda s: None)
    assert again == ref
    # another model / length: recomputed (and the cache replaced)
    monkeypatch.undo()
    other = dict(meta, base="c")
    new = dpo.reference_logps(model, rows[:1], path, other,
                              logf=lambda s: None)
    assert set(new) == {rows[0]["key"]}
    assert json.load(open(path))["meta"] == other


def test_steps_raise_the_margin_on_a_toy_pair(tiny):
    torch = pytest.importorskip("torch")
    model = peft_model(tiny, seed=1)
    rows = rows_for(tiny, 1)
    ref = dpo.reference_logps(model, rows, None, {}, logf=lambda s: None)
    ds = dpo.PairRows(rows, ["k"], ref)
    o = dpo.parse(["train", "--data", "x", "--out", "y"])
    o.beta, o.sft_alpha, o.length_norm = 0.5, 0.0, False
    opt = torch.optim.AdamW([p for p in model.parameters()
                             if p.requires_grad], lr=5e-2)
    margins = []
    for _ in range(8):
        inputs = dpo.collate([ds[0]])
        loss, st = dpo.pair_loss(model, inputs, o)
        margins.append(st["margin"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert margins[0] == pytest.approx(0.0, abs=1e-3)       # policy == ref
    assert margins[-1] > margins[0] + 0.05
    # no precomputed reference: the same margin from the adapter-off forward
    nan = dpo.PairRows(rows, ["k"], None)
    with torch.no_grad():
        _l, pre = dpo.pair_loss(model, dpo.collate([ds[0]]), o)
        _l, onfly = dpo.pair_loss(model, dpo.collate([nan[0]]), o)
    assert pre["margin"] > margins[0]
    assert onfly["margin"] == pytest.approx(pre["margin"], abs=1e-3)


def test_a_whole_run_resume_and_merge(tiny, tmp_path):
    fake_sft_run(tmp_path / "sft")
    write_pairs(tmp_path / "pairs" / "dpo.jsonl", 24, sessions=6)
    a = argv_of(tmp_path, "--no-4bit", "--max-steps", "3", "--accum", "2",
                "--lr", "1e-3", "--eval-steps", "2", "--save-steps", "2",
                "--log-steps", "1")
    a[a.index("--base") + 1] = tiny
    s = dpo.train(dpo.parse(a))
    out = tmp_path / "run"
    assert os.path.exists(out / "adapter" / "adapter_config.json")
    assert os.path.exists(out / "ref_logps.json")
    assert "eval_accuracy" in s["metrics"] and "eval_loss" in s["metrics"]
    assert s["base"] == tiny and s["family"] == "qwen3" and s["beta"] == 0.1
    tr = json.load(open(out / "train.json"))
    assert tr["adapter"] == str(out / "adapter")
    ck = sft.last_checkpoint(str(out))
    assert ck and ck.endswith("checkpoint-3")
    # other data: no resume
    write_pairs(tmp_path / "pairs" / "dpo.jsonl", 26, sessions=6)
    with pytest.raises(SystemExit, match="other data"):
        dpo.train(dpo.parse(a))
    # merged into the base; the run's commands are SFT's
    m = sft.merge(sft.parse(["merge", "--run", str(out), "--dtype",
                             "float32", "--device", "cpu"]))
    assert os.path.exists(os.path.join(m, "config.json"))
    cmds = sft.gguf(sft.parse(["gguf", "--run", str(out), "--name",
                               "tsk-q8r3d"]))
    assert cmds[0][-1] == "f16" and "tsk-q8r3d-f16.gguf" in cmds[0][4]
    mf = sft.write_modelfile(sft.parse(["modelfile", "--run", str(out),
                                        "--name", "tsk-q8r3d"]))
    assert "tsk-q8r3d-Q4_K_M.gguf" in open(mf).read()
