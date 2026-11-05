#!/usr/bin/env python3
"""
DPO of the student (docs/embodiment.md, "DPO training"), on the project's
own trainer code (flow.train.sft) and one 24 GB GPU; no TRL (the pinned
transformers is 5.x, and the dependency set stays as it is).

  start     the SFT run's merged model (<sft run>/merged, fp16: `train_sft.py
            merge`), loaded in 4 bit (NF4) like SFT, with a NEW LoRA on the
            same projections (rank 16, alpha 32).
  reference the same model with the LoRA switched off (peft's
            disable_adapter()) = the SFT student, exactly; no second copy in
            memory. --precompute-ref [default]: its summed reply log-probs
            for every pair are computed once before training (adapter off,
            no grad) and kept in <out>/ref_logps.json (keyed by the pair's
            token ids and the base's files, so a rerun, a resume or a
            changed beta reuses them), so a step is one forward per side.
            --no-precompute-ref computes them in the step instead.
  data      dpo.jsonl of scripts/dpo_pairs.py (TRL's conversational format:
            prompt messages incl. system and history, chosen / rejected
            assistant messages) with dpo.meta.jsonl beside it for the kinds
            (look_flip ...) and sessions. --data a.jsonl:1,b.jsonl:2 mixes
            files; the number is a weight (a pair is repeated that many
            times: 2 twice; a fraction is a chance). Pairs whose chosen and
            rejected are the same text are dropped. Held out by session
            (--eval-frac; flow.train.sft's hash rule with its own seed) for
            the eval accuracy and margins.
            With --sft-run its split.json is copied to <out>/split.json (the
            eval script reads it: the same held-out sessions as the SFT run,
            so the comparison is fair) and the pairs of ITS held-out sessions
            are left out of training (counted; --allow-eval-sessions to keep
            them): build the pairs with `dpo_pairs.py --side train`.
  tokens    exactly SFT's: flow.train.chat for the family (the empty think
            block on every assistant turn), each template segment tokenized
            on its own, labels only on the reply and its end-of-turn token
            (the history's replies get "weight": 0, as a per-turn export's
            history does). The prompt is cut from the left only when a pair
            is over --max-len: whole oldest history turns go (the system
            prompt and the user line stay, the replies are never cut); a
            pair that still does not fit is left out. Both counted.
  loss      -log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)]) on the
            SUMMED reply log-probs (--length-norm: per token), plus
            --sft-alpha times the chosen reply's mean NLL (RPO), which keeps
            it near the SFT distribution. Logged: loss, rewards/chosen,
            rewards/rejected, margin, accuracy (the implicit reward prefers
            chosen), nll, and accuracy / margin per pair kind.
  run       batch 1 pair, gradient accumulation (--accum 8), lr 5e-6 cosine
            with warmup, gradient checkpointing, paged 8-bit AdamW, bf16;
            checkpoints and resume as SFT (data.json holds the hash of the
            tokenized pairs; a changed set stops the run unless --restart).
            The first steps log tokens/s and peak memory. The adapter goes
            to <out>/adapter.
  after     train.json has base = the SFT merged model and the DPO adapter,
            so merge / gguf / modelfile are flow.train.sft's own, unchanged
            (`merge` merges the adapter into the SFT merged model ->
            <out>/merged).

Usage (scripts/train_dpo.py is the same entry point):
    python scripts/train_dpo.py stats --sft-run runs/sft/q8r3 --data runs/dpo/q8r3/dpo.jsonl
    python scripts/train_dpo.py train --sft-run runs/sft/q8r3 --data runs/dpo/q8r3/dpo.jsonl --out runs/dpo/q8r3d
    python scripts/train_dpo.py merge --run runs/dpo/q8r3d
    python scripts/train_dpo.py gguf --run runs/dpo/q8r3d --llama-cpp ~/llama.cpp --name tsk-q8r3d
    python scripts/train_dpo.py modelfile --run runs/dpo/q8r3d --name tsk-q8r3d --create
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict

import numpy as np

try:
    from . import chat
    from . import sft
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from flow.train import chat
    from flow.train import sft

SPLIT_SEED = 1
REF_VERSION = 1


def log(s):
    print(s, flush=True)


# ---------------------------------------------------------------- the loss

def _logsigmoid(x):
    """log sigmoid(x) for a float or a torch tensor."""
    if isinstance(x, (int, float)):
        return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(
            math.exp(x))
    import torch.nn.functional as F
    return F.logsigmoid(x)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, n_c=1, n_r=1, beta=0.1,
             sft_alpha=0.0, length_norm=False):
    """(loss, stats) of one pair from the summed reply log-probs of the
    policy (pi_*) and the reference (ref_*) and the reply token counts
    (n_*); floats or 0-d tensors (module docstring). stats: rewards/chosen,
    rewards/rejected (beta * the log-ratio), margin, accuracy (1 / 0.5 on a
    tie / 0), nll (the chosen reply's mean NLL)."""
    d_c, d_r = pi_c - ref_c, pi_r - ref_r
    if length_norm:
        d_c, d_r = d_c / n_c, d_r / n_r
    rc, rr = beta * d_c, beta * d_r
    margin = rc - rr
    nll = -pi_c / n_c
    loss = -_logsigmoid(margin)
    if sft_alpha:
        loss = loss + sft_alpha * nll
    m = float(margin)
    stats = {"rewards/chosen": float(rc), "rewards/rejected": float(rr),
             "margin": m, "accuracy": 1.0 if m > 0 else 0.5 if m == 0
             else 0.0, "nll": float(nll)}
    return loss, stats


# -------------------------------------------------------------------- data

def parse_specs(spec) -> list:
    """[(path, weight)] from "a.jsonl:1,b.jsonl:2" (no :N is weight 1; a
    folder is its dpo.jsonl)."""
    out = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        head, sep, tail = item.rpartition(":")
        path, w = item, 1.0
        if sep and head and re.fullmatch(r"\d+(?:\.\d+)?", tail):
            path, w = head, float(tail)
        if w < 0:
            raise SystemExit(f"--data {item!r}: negative weight")
        out.append((path, w))
    if not out:
        raise SystemExit("--data: no file")
    return out


def pairs_path(path) -> str:
    return os.path.join(path, "dpo.jsonl") if os.path.isdir(path) else path


def meta_path(path) -> str:
    d, b = os.path.split(pairs_path(path))
    stem = b[:-len(".jsonl")] if b.endswith(".jsonl") else b
    return os.path.join(d, stem + ".meta.jsonl")


def _reply(x) -> str:
    if isinstance(x, list):
        return x[-1]["content"] if x else ""
    return x or ""


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_pairs(path, weight=1.0) -> tuple:
    """([pair], info) from one dpo.jsonl (+ its dpo.meta.jsonl): pair
    {prompt (messages), chosen, rejected (texts), kind, avatar, id, session,
    weight, meta {session} (for sft.split), src}; info counts what was left
    out."""
    p = pairs_path(path)
    rows = _jsonl(p)
    mp = meta_path(path)
    metas = _jsonl(mp) if os.path.exists(mp) else []
    info = {"rows": len(rows), "same_text": 0, "no_meta": 0}
    if metas and len(metas) != len(rows):
        log(f"dpo: {mp} has {len(metas)} lines, {p} {len(rows)}: the meta "
            "is ignored")
        metas = []
    if not metas:
        info["no_meta"] = len(rows)
    out = []
    for i, r in enumerate(rows):
        m = metas[i] if metas else {}
        chosen, rejected = _reply(r["chosen"]), _reply(r["rejected"])
        if chosen == rejected:
            info["same_text"] += 1
            continue
        prompt = [{"role": x["role"], "content": x.get("content") or ""}
                  for x in r["prompt"]]
        rid = m.get("id")
        if rid:
            session = rid.rsplit("/", 1)[0]
        else:
            first = next((x["content"] for x in prompt
                          if x["role"] != "system"), "")
            session = "p" + hashlib.sha256(first.encode("utf-8")).hexdigest()[:10]
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                    "kind": m.get("kind") or "unknown",
                    "avatar": m.get("avatar") or "?", "id": rid,
                    "session": session, "weight": weight,
                    "meta": {"session": session},
                    "src": os.path.basename(os.path.dirname(
                        os.path.abspath(p))) + "/" + os.path.basename(p)})
    return out, info


def expand(pairs, seed=0) -> list:
    """The pairs repeated by their weights (a fraction is a chance, seeded)."""
    rng = random.Random(seed)
    out = []
    for p in pairs:
        w = p["weight"]
        n = int(w) + (1 if rng.random() < w - int(w) else 0)
        out += [p] * n
    return out


def split_pairs(pairs, eval_frac, seed=SPLIT_SEED, sft_eval=None):
    """(train, eval, info): by session (flow.train.sft.split's rule, its own
    seed). The sessions in `sft_eval` (the SFT run's held-out ones) go to
    neither side."""
    sft_eval = set(sft_eval or ())
    kept = [p for p in pairs if p["session"] not in sft_eval]
    info = {"sft_eval_dropped": len(pairs) - len(kept)}
    if not kept:
        return [], [], info
    train, ev, sp = sft.split(kept, eval_frac, seed, by_row=True)
    info.update(train_sessions=sp["train_sessions"],
                eval_sessions=sp["eval_sessions"],
                rule=sp["rule"], eval_frac=eval_frac, seed=seed)
    return train, ev, info


# ------------------------------------------------------------------ tokens

def _side(tok, prompt, reply, family):
    """(ids, labels) of prompt + reply as SFT encodes the same turn: the
    prompt's assistant turns (the history) are not learned."""
    msgs = [dict(m, weight=0) if m["role"] == "assistant" else dict(m)
            for m in prompt] + [{"role": "assistant", "content": reply}]
    r = sft.encode(tok, sft.clean({"messages": msgs}), family)
    return r["input_ids"], r["labels"]


def pair_key(c_ids, r_ids) -> str:
    h = hashlib.sha256()
    for ids in (c_ids, r_ids):
        h.update(np.asarray(ids, np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()[:20]


def n_labelled(labels) -> int:
    """The tokens a log-prob is summed over (the first token has no
    logit before it, as in masked_loss)."""
    return int(sum(1 for x in labels[1:] if x != -100))


def encode_pair(tok, pair, family, max_len=None):
    """(row | None, history turns cut). The row: chosen / rejected ids and
    labels, key, kind. When the longer side is over max_len the oldest
    history turn (a user line and its reply) goes, again and again; the
    system prompt, the last user line and the two replies stay whole. None
    when that is not enough (or there is no history to give)."""
    prompt = [dict(m) for m in pair["prompt"]]
    cut = 0
    while True:
        c_ids, c_lab = _side(tok, prompt, pair["chosen"], family)
        r_ids, r_lab = _side(tok, prompt, pair["rejected"], family)
        if not max_len or max(len(c_ids), len(r_ids)) <= max_len:
            break
        j = next((i for i, m in enumerate(prompt) if m["role"] ==
                  "assistant"), None)
        if j is None:
            return None, cut
        start = 1 if prompt and prompt[0]["role"] == "system" else 0
        del prompt[start:j + 1]
        cut += 1
    if not n_labelled(c_lab) or not n_labelled(r_lab):
        return None, cut
    return {"chosen_ids": c_ids, "chosen_labels": c_lab,
            "rejected_ids": r_ids, "rejected_labels": r_lab,
            "key": pair_key(c_ids, r_ids), "kind": pair["kind"],
            "avatar": pair["avatar"], "cut": cut}, cut


def encode_all(tok, pairs, family, max_len):
    """(rows, stats): stats {dropped, cut_pairs, cut_turns}."""
    rows, st = [], Counter()
    memo = {}
    for p in pairs:
        k = id(p)
        if k not in memo:
            row, cut = encode_pair(tok, p, family, max_len)
            memo[k] = (row, cut)
            if cut:
                st["cut_pairs"] += 1
                st["cut_turns"] += cut
            if row is None:
                st["dropped"] += 1
        row, _cut = memo[k]
        if row is not None:
            rows.append(row)
    return rows, dict(st)


def rows_hash(rows, max_len=None) -> str:
    """The hash of the tokenized pairs (both sides' ids and labels, in
    order)."""
    flat = []
    for r in rows:
        flat.append({"input_ids": r["chosen_ids"],
                     "labels": r["chosen_labels"]})
        flat.append({"input_ids": r["rejected_ids"],
                     "labels": r["rejected_labels"]})
    return sft.data_hash(flat)


def _dist(xs):
    if not len(xs):
        return {"n": 0, "mean": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    return {"n": len(xs), "mean": round(float(np.mean(xs)), 1),
            "p50": sft._pct(xs, 50), "p90": sft._pct(xs, 90),
            "p99": sft._pct(xs, 99), "max": int(max(xs))}


def length_stats(tok, pairs, family, max_len=None) -> dict:
    """Token counts over `pairs` (untruncated): the prompt, each reply,
    each whole side; how many pairs are over `max_len` (the history would
    be cut) and how many cannot fit even then."""
    prompt, chosen, rejected, whole = [], [], [], []
    pairs = list({id(p): p for p in pairs}.values())     # weights repeat
    for p in pairs:
        c_ids, c_lab = _side(tok, p["prompt"], p["chosen"], family)
        r_ids, r_lab = _side(tok, p["prompt"], p["rejected"], family)
        first = next((i for i, x in enumerate(c_lab) if x != -100), 0)
        prompt.append(first)
        chosen.append(len(c_ids) - first)
        rejected.append(len(r_ids) - next(
            (i for i, x in enumerate(r_lab) if x != -100), 0))
        whole.append(max(len(c_ids), len(r_ids)))
    out = {"pairs": len(pairs), "prompt": _dist(prompt),
           "chosen": _dist(chosen), "rejected": _dist(rejected),
           "longest side": _dist(whole),
           "auto_max_len": sft.auto_max_len(whole)}
    if max_len:
        over = [i for i, w in enumerate(whole) if w > max_len]
        out["max_len"] = max_len
        out["over_max_len"] = len(over)
        # Cutting all the history leaves system + last user + the reply.
        out["cannot_fit"] = sum(
            1 for i in over if _floor_len(tok, pairs[i], family) > max_len)
    return out


def _floor_len(tok, pair, family):
    """The longer side with the history gone (the shortest a pair gets)."""
    prompt = [m for m in pair["prompt"] if m["role"] == "system"] + \
        [pair["prompt"][-1]]
    return max(len(_side(tok, prompt, pair["chosen"], family)[0]),
               len(_side(tok, prompt, pair["rejected"], family)[0]))


def length_text(rep) -> str:
    out = [f"tokens over {rep['pairs']} pairs (untruncated):"]
    for k in ("prompt", "chosen", "rejected", "longest side"):
        d = rep[k]
        out.append(f"  {k:13s} mean {d['mean']:8.1f}  p50 {d['p50']:6d}  "
                   f"p90 {d['p90']:6d}  p99 {d['p99']:6d}  max {d['max']:6d}")
    out.append(f"  --max-len auto: {rep['auto_max_len']}")
    if "max_len" in rep:
        out.append(f"  at --max-len {rep['max_len']}: {rep['over_max_len']} "
                   f"pairs over (their history is cut), {rep['cannot_fit']} "
                   "of them cannot fit even without history (left out)")
    return "\n".join(out)


def counts_text(pairs, title="pairs") -> str:
    kinds = Counter(p["kind"] for p in pairs)
    avs = Counter(p["avatar"] for p in pairs)
    sess = len({p["session"] for p in pairs})
    return "\n".join([
        f"{title}: {len(pairs)} in {sess} sessions",
        "  by kind: " + (", ".join(f"{k} {v}" for k, v in sorted(
            kinds.items())) or "-"),
        "  by avatar: " + (", ".join(f"{k} {v}" for k, v in sorted(
            avs.items())) or "-")])


# --------------------------------------------------------------- reference

def base_fingerprint(base) -> str:
    """What identifies the base model's weights: its folder's config and
    safetensors names and sizes (a name or a path that is no folder:
    itself)."""
    if not os.path.isdir(base):
        return str(base)
    parts = [os.path.abspath(base)]
    for f in sorted(os.listdir(base)):
        if f == "config.json" or f.endswith(".safetensors"):
            parts.append(f"{f}:{os.path.getsize(os.path.join(base, f))}")
    return "|".join(parts)


def seq_logp(model, ids, labels, keep_logits=True):
    """(summed log-prob, token count) of the labelled tokens of one
    sequence (ids, labels: (1, L) long tensors); the head runs only where a
    label is (logits_to_keep), as sft.masked_loss."""
    import torch
    import torch.nn.functional as F
    tgt = labels[0, 1:]
    pos = (tgt != -100).nonzero().squeeze(-1)
    mask = torch.ones_like(ids)
    if keep_logits:
        out = model(input_ids=ids, attention_mask=mask, logits_to_keep=pos)
        logits = out.logits[0]
        if logits.shape[0] != pos.numel():
            logits = logits[pos]
    else:
        logits = model(input_ids=ids, attention_mask=mask).logits[0, :-1][pos]
    lp = -F.cross_entropy(logits.float(), tgt[pos], reduction="none")
    return lp.sum(), int(pos.numel())


def _tensors(row, device):
    import torch
    out = {}
    for k in ("chosen", "rejected"):
        out[k] = (torch.tensor([row[f"{k}_ids"]], dtype=torch.long,
                               device=device),
                  torch.tensor([row[f"{k}_labels"]], dtype=torch.long,
                               device=device))
    return out


def policy_logps(model, row, device, keep_logits=True) -> tuple:
    t = _tensors(row, device)
    c, _ = seq_logp(model, *t["chosen"], keep_logits)
    r, _ = seq_logp(model, *t["rejected"], keep_logits)
    return c, r


def reference_logps(model, rows, cache_path, meta, probe=50, keep_logits=True,
                    logf=log) -> dict:
    """{pair key: [chosen sum, rejected sum]} of the reference (the model
    with its adapter off, no grad) for every row, from the cache file when
    its header equals `meta` and computed for what it lacks; saved back."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("meta") == meta:
            cache = {k: list(v) for k, v in old.get("logps", {}).items()}
        else:
            logf("dpo: the reference cache is for another model or length; "
                 "recomputing")
    todo = []
    seen = set()
    for r in rows:
        if r["key"] not in cache and r["key"] not in seen:
            seen.add(r["key"])
            todo.append(r)
    if todo:
        import torch
        dev = model.device
        was = model.training
        model.eval()
        t0 = time.monotonic()
        logf(f"dpo: reference log-probs of {len(todo)} pairs "
             f"({len(cache)} cached)")
        with torch.no_grad(), model.disable_adapter():
            for i, r in enumerate(todo, 1):
                c, rj = policy_logps(model, r, dev, keep_logits)
                cache[r["key"]] = [float(c), float(rj)]
                if i % probe == 0:
                    el = time.monotonic() - t0
                    logf(f"dpo: reference {i}/{len(todo)} "
                         f"({el / i:.2f} s a pair)")
        model.train(was)
        if cache_path:
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"meta": meta, "logps": cache}, fh)
            os.replace(tmp, cache_path)
    return cache


# ----------------------------------------------------------------- dataset

class PairRows:
    """A torch-style dataset over encoded pairs (no torch import here);
    `kinds` the list that kind ids index, `ref` {key: [c, r]} or None."""

    def __init__(self, rows, kinds, ref=None):
        self.rows, self.kinds, self.ref = rows, kinds, ref

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ref = (self.ref or {}).get(r["key"])
        return {"chosen_ids": r["chosen_ids"],
                "chosen_labels": r["chosen_labels"],
                "rejected_ids": r["rejected_ids"],
                "rejected_labels": r["rejected_labels"],
                "ref": list(ref) if ref else [float("nan")] * 2,
                "kind": self.kinds.index(r["kind"])}


def collate(batch):
    import torch
    if len(batch) != 1:
        raise ValueError("batch size 1 (one pair a step; gradient "
                         "accumulation does the batching)")
    b = batch[0]
    out = {k: torch.tensor([b[k]], dtype=torch.long)
           for k in ("chosen_ids", "chosen_labels", "rejected_ids",
                     "rejected_labels")}
    out["ref"] = torch.tensor([b["ref"]], dtype=torch.float64)
    out["kind"] = torch.tensor([b["kind"]], dtype=torch.long)
    return out


def pair_loss(model, inputs, o, keep_logits=True):
    """(loss, stats) of one collated pair. The reference comes with the
    batch (precomputed) or, if it is NaN, from the model with its adapter
    off."""
    import torch
    pi_c, n_c = seq_logp(model, inputs["chosen_ids"],
                         inputs["chosen_labels"], keep_logits)
    pi_r, n_r = seq_logp(model, inputs["rejected_ids"],
                         inputs["rejected_labels"], keep_logits)
    ref = inputs["ref"][0]
    if torch.isnan(ref).any():
        with torch.no_grad(), model.disable_adapter():
            ref_c, _ = seq_logp(model, inputs["chosen_ids"],
                                inputs["chosen_labels"], keep_logits)
            ref_r, _ = seq_logp(model, inputs["rejected_ids"],
                                inputs["rejected_labels"], keep_logits)
    else:
        ref_c, ref_r = ref[0].to(pi_c.dtype), ref[1].to(pi_r.dtype)
    return dpo_loss(pi_c, pi_r, ref_c, ref_r, n_c, n_r, o.beta,
                    o.sft_alpha, o.length_norm)


class Means:
    """Running means of named values since the last reset."""

    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, k, v):
        self.s[k] += v
        self.n[k] += 1

    def means(self, prefix="") -> dict:
        return {prefix + k: round(self.s[k] / self.n[k], 5) for k in self.s}

    def reset(self):
        self.s.clear()
        self.n.clear()


def note(means, stats, kind):
    for k, v in stats.items():
        means.add(k, v)
        if k in ("accuracy", "margin"):
            means.add(f"{k}/{kind}", v)


def make_trainer(model, tok, args, train_ds, eval_ds, callbacks, o):
    import torch
    from transformers import Trainer

    class DPOTrainer(Trainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # A per-pair loss: the Trainer scales it by the accumulation
            # steps itself.
            self.model_accepts_loss_kwargs = False
            self.train_means, self.eval_means = Means(), Means()

        def _loss(self, model, inputs, means):
            loss, stats = pair_loss(model, inputs, o)
            kind = train_ds.kinds[int(inputs["kind"][0])]
            note(means, dict(stats, loss=float(loss)), kind)
            return loss

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            loss = self._loss(model, inputs, self.train_means)
            return (loss, {}) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only,
                            ignore_keys=None):
            with torch.no_grad():
                loss = self._loss(model, inputs, self.eval_means)
            return loss.detach(), None, None

        def log(self, logs, *a, **k):
            logs = dict(logs)
            if any(x.startswith("eval_") for x in logs):
                logs.update(self.eval_means.means("eval_"))
            elif "loss" in logs:
                logs.update(self.train_means.means())
                self.train_means.reset()
            return super().log(logs, *a, **k)

        def evaluate(self, *a, **k):
            self.eval_means.reset()
            out = super().evaluate(*a, **k)
            out.update(self.eval_means.means("eval_"))
            return out

    kw = {"model": model, "args": args, "train_dataset": train_ds,
          "eval_dataset": eval_ds, "data_collator": collate,
          "callbacks": callbacks}
    import inspect
    kw["processing_class" if "processing_class" in inspect.signature(
        Trainer.__init__).parameters else "tokenizer"] = tok
    return DPOTrainer(**sft._filter_kwargs(Trainer.__init__, kw, "Trainer"))


# --------------------------------------------------------------- the plan

def sft_info(run) -> dict:
    """The SFT run's train.json, data.json and split.json that matter."""
    out = {"run": run}
    for name in ("train", "data", "split"):
        p = os.path.join(run, f"{name}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[name] = json.load(fh)
    return out


def resolve(o):
    """Fill o.base, o.family, o.sft from --sft-run / --base / --template."""
    o.sft = sft_info(o.sft_run) if o.sft_run else {}
    if not o.base:
        if not o.sft_run:
            raise SystemExit("dpo: give --sft-run (its merged model is the "
                             "start) or --base")
        o.base = os.path.join(o.sft_run, "merged")
    fam = o.template or (o.sft.get("train") or {}).get("family")
    o.family = fam or chat.family_for(o.base)
    return o


def load_data(o):
    """(pairs, info) over the --data specs, weights not yet applied."""
    pairs, info = [], Counter()
    for path, w in parse_specs(o.data):
        if not os.path.exists(pairs_path(path)):
            raise SystemExit(f"dpo: no {pairs_path(path)}")
        got, i = load_pairs(path, w)
        log(f"dpo: {pairs_path(path)}: {len(got)} pairs"
            + (f" x{w:g}" if w != 1 else "")
            + (f" ({i['same_text']} with chosen == rejected left out)"
               if i["same_text"] else "")
            + ("" if not i["no_meta"] else " (no meta: kinds unknown)"))
        pairs += got
        info.update(i)
    return pairs, dict(info)


def prepare(o, tok):
    """Pairs -> split -> rows (counted); the plan of a run."""
    pairs, info = load_data(o)
    sft_eval = None
    if o.sft and o.sft.get("split") and not o.allow_eval_sessions:
        sft_eval = o.sft["split"].get("eval_sessions")
    train, ev, sp = split_pairs(pairs, o.eval_frac, o.split_seed, sft_eval)
    if sp["sft_eval_dropped"]:
        log(f"dpo: {sp['sft_eval_dropped']} pairs are from the SFT run's "
            "held-out sessions: left out of training and eval, so the "
            "comparison with the SFT student stays fair")
    if not train:
        raise SystemExit(
            "dpo: no training pairs" + (
                " (every pair is from the SFT run's held-out sessions: "
                "build them on the training sessions, `dpo_pairs.py "
                "--split <sft run>/split.json --side train`, or "
                "--allow-eval-sessions)" if sp["sft_eval_dropped"] else ""))
    log(counts_text(train, "train") + "\n" + counts_text(ev, "eval"))
    train = expand(train, o.seed)
    return train, ev, sp, info


def plan(o, tok):
    """Everything before the model loads: pairs, rows, the hash, the
    resume check, data.json. Returns a dict."""
    train, ev, sp, info = prepare(o, tok)
    rep = length_stats(tok, train + ev, o.family)
    log(length_text(rep))
    o.max_len_arg = o.max_len
    o.max_len = rep["auto_max_len"] if o.max_len == "auto" else int(o.max_len)
    rows_tr, st = encode_all(tok, train, o.family, o.max_len)
    rows_ev, st_e = encode_all(tok, ev, o.family, o.max_len)
    log(f"dpo: {len(rows_tr)} train pairs ({st.get('cut_pairs', 0)} with "
        f"their oldest history cut to fit {o.max_len} tokens, "
        f"{st.get('dropped', 0)} left out: too long), {len(rows_ev)} eval "
        f"pairs ({st_e.get('cut_pairs', 0)}, {st_e.get('dropped', 0)})")
    if not rows_tr:
        raise SystemExit("dpo: nothing to train on")
    ex = {"messages": (train[0]["prompt"]
                       + [{"role": "assistant", "content": train[0]["chosen"]}])}
    probs = sft.check_template(tok, o.family, sft.clean(ex))
    if probs:
        for p in probs:
            log(f"dpo: TEMPLATE MISMATCH {p}")
        if not o.allow_template_mismatch:
            raise SystemExit("dpo: the tokenizer's chat template and "
                             f"flow.train.chat {o.family} disagree (above); "
                             "--template, or --allow-template-mismatch")
    dh = rows_hash(rows_tr + rows_ev)
    os.makedirs(o.out, exist_ok=True)
    meta = os.path.join(o.out, "data.json")
    ck = sft.last_checkpoint(o.out)
    if ck and os.path.exists(meta):
        with open(meta, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("hash") != dh and not o.restart:
            raise SystemExit(f"dpo: {o.out} has checkpoints for other data "
                             f"(hash {old.get('hash')} vs {dh}); --restart "
                             "to start over, or another --out")
    if o.restart and ck:
        shutil.rmtree(os.path.join(o.out, "checkpoints"))
        ck = None
    sdata = (o.sft.get("data") or {}) if o.sft else {}
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"hash": dh, "train_rows": len(rows_tr),
                   "eval_rows": len(rows_ev), "max_len": o.max_len,
                   "family": o.family, "base": o.base,
                   "pairs": [list(x) for x in parse_specs(o.data)],
                   "sft_run": o.sft_run,
                   "data": o.eval_data or sdata.get("data")}, fh, indent=1)
    # The eval script reads split.json: the SFT run's, as it was.
    if o.sft.get("split"):
        with open(os.path.join(o.out, "split.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(o.sft["split"], fh, indent=1)
    with open(os.path.join(o.out, "dpo_split.json"), "w",
              encoding="utf-8") as fh:
        json.dump({k: sp[k] for k in sp if k != "sft_eval_dropped"}
                  | {"sft_eval_dropped": sp["sft_eval_dropped"]}, fh, indent=1)
    with open(os.path.join(o.out, "lengths.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return {"rows_tr": rows_tr, "rows_ev": rows_ev, "hash": dh, "ck": ck,
            "rep": rep, "kinds": sorted({r["kind"] for r in rows_tr
                                          + rows_ev}),
            "pairs": len(train) + len(ev)}


# -------------------------------------------------------------------- train

def train(o) -> dict:
    resolve(o)
    tok0 = sft.load_tokenizer(o.base)
    pl = plan(o, tok0)
    rows_tr, rows_ev, ck = pl["rows_tr"], pl["rows_ev"], pl["ck"]
    o.backend_used = "peft"
    model, tok, used = sft.load_for_training(o)
    log(f"dpo: {used}: {o.base}, LoRA r {o.rank} alpha {o.alpha}, "
        f"{'4-bit' if o.load_4bit else 'unquantized'}; beta {o.beta}, "
        f"sft-alpha {o.sft_alpha}, lr {o.lr}, accum {o.accum}"
        + (", length-normalised" if o.length_norm else ""))
    try:
        model.print_trainable_parameters()
    except Exception:  # noqa: BLE001 - a report
        pass
    ref = None
    if o.precompute_ref:
        meta = {"version": REF_VERSION, "base": base_fingerprint(o.base),
                "family": o.family, "max_len": o.max_len}
        ref = reference_logps(model, rows_tr + rows_ev,
                              os.path.join(o.out, "ref_logps.json"), meta,
                              logf=log)
    check_start(model, rows_tr[0], ref)
    kinds = pl["kinds"]
    o.eval_n = len(rows_ev)
    args = sft.training_args(o, len(rows_tr))
    sizes = [{"input_ids": r["chosen_ids"] + r["rejected_ids"]}
             for r in rows_tr]
    tr = make_trainer(model, tok, args, PairRows(rows_tr, kinds, ref),
                      PairRows(rows_ev, kinds, ref) if rows_ev else None,
                      [sft.speed_callback(sizes, o.accum, o.probe_steps)], o)
    if ck:
        log(f"dpo: resuming from {ck}")
    t0 = time.monotonic()
    res = tr.train(resume_from_checkpoint=ck)
    hours = (time.monotonic() - t0) / 3600
    metrics = dict(res.metrics)
    if rows_ev:
        metrics.update(tr.evaluate())
    adir = os.path.join(o.out, "adapter")
    tr.save_model(adir)
    sft.save_tokenizer(tok0, adir, o.family)
    summary = {"base": o.base, "family": o.family, "backend": used,
               "rank": o.rank, "alpha": o.alpha, "max_len": o.max_len,
               "load_4bit": o.load_4bit, "epochs": o.epochs, "lr": o.lr,
               "accum": o.accum, "beta": o.beta, "sft_alpha": o.sft_alpha,
               "length_norm": o.length_norm,
               "precompute_ref": o.precompute_ref,
               "hours": round(hours, 3),
               "metrics": {k: (round(v, 5) if isinstance(v, float) else v)
                           for k, v in metrics.items()},
               "lengths": pl["rep"], "data_hash": pl["hash"],
               "sft_run": o.sft_run, "versions": sft.versions(),
               "adapter": adir}
    with open(os.path.join(o.out, "train.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    log(f"dpo: adapter -> {adir}; eval accuracy "
        f"{metrics.get('eval_accuracy')}, margin "
        f"{metrics.get('eval_margin')}; {hours:.2f} h")
    return summary


def check_start(model, row, ref, tol=0.5, logf=log):
    """Before step 1 the policy (LoRA B = 0) must be the reference: the
    difference of the first pair's log-probs (summed) is printed, a warning
    over `tol` nats. Returns it, or None without a precomputed reference."""
    if not ref or row["key"] not in ref:
        return None
    import torch
    was = model.training
    model.eval()
    with torch.no_grad():
        c, r = policy_logps(model, row, model.device)
    model.train(was)
    d = max(abs(float(c) - ref[row["key"]][0]),
            abs(float(r) - ref[row["key"]][1]))
    logf(f"dpo: step 0: |policy - reference| on the first pair {d:.4f} "
         "nats (must be ~0)")
    if d > tol:
        logf("dpo: WARNING: the policy differs from the reference before "
             "training: check the base, the dtype and the tokenization "
             "against the run that made ref_logps.json")
    return d


def stats(o) -> dict:
    resolve(o)
    tok = sft.load_tokenizer(o.base)
    pairs, info = load_data(o)
    sft_eval = None
    if o.sft.get("split") and not o.allow_eval_sessions:
        sft_eval = o.sft["split"].get("eval_sessions")
    train_p, ev_p, sp = split_pairs(pairs, o.eval_frac, o.split_seed, sft_eval)
    print(counts_text(pairs, "all pairs"))
    if sp["sft_eval_dropped"]:
        print(f"{sp['sft_eval_dropped']} pairs are from the SFT run's "
              "held-out sessions (not trained on)")
    print(counts_text(train_p, "train") + "\n" + counts_text(ev_p, "eval"))
    ml = None if o.max_len == "auto" else int(o.max_len)
    rep = length_stats(tok, train_p + ev_p, o.family, ml)
    print(length_text(rep))
    if o.json:
        with open(o.json, "w", encoding="utf-8") as fh:
            json.dump({"rep": rep, "load": info}, fh, indent=1)
    return rep


# ----------------------------------------------------------------- cli

def add_model_args(p):
    p.add_argument("--sft-run", default=None,
                   help="the SFT run (train_sft.py train --out ...) whose "
                        "merged model is the start; its split.json is the "
                        "eval's")
    p.add_argument("--base", default=None,
                   help="[<sft run>/merged] the model to start from")
    p.add_argument("--template", default=None, choices=chat.FAMILIES,
                   help="chat template family [the SFT run's]")
    p.add_argument("--data", required=True,
                   help="dpo.jsonl (or its folder), several with weights: "
                        "a.jsonl:1,b.jsonl:2")
    p.add_argument("--eval-frac", type=float, default=0.1,
                   help="share of the sessions held out for the eval "
                        "accuracy [0.1]")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--allow-eval-sessions", action="store_true",
                   help="train on pairs from the SFT run's held-out sessions "
                        "too (the comparison is then not fair)")
    p.add_argument("--max-len", default="auto",
                   help="tokens per side, or auto (p99, rounded up to 256); "
                        "over it the oldest history turns are cut")


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats", help="pairs by kind / avatar, token lengths, "
                                     "how many would be cut")
    add_model_args(p)
    p.add_argument("--json", default=None)
    p = sub.add_parser("train")
    add_model_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--eval-data", default=None,
                   help="the flow.dataset folder eval_sft.sh scores on "
                        "[the SFT run's]")
    p.add_argument("--backend", default="peft", choices=("peft",))
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft-alpha", type=float, default=0.2,
                   help="weight of the chosen reply's NLL added to the loss "
                        "(RPO) [0.2]; 0 is plain DPO")
    p.add_argument("--length-norm", action="store_true",
                   help="log-ratios per token, not summed")
    p.add_argument("--precompute-ref", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="reference log-probs once before training "
                        "(ref_logps.json) [on]")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="stop after this many optimizer steps (a probe)")
    p.add_argument("--accum", type=int, default=8,
                   help="pairs per optimizer step")
    p.add_argument("--warmup", type=float, default=0.1,
                   help="share of the steps warming up")
    p.add_argument("--eval-steps", type=int, default=25)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--keep", type=int, default=3, help="checkpoints kept")
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--probe-steps", type=int, default=10,
                   help="steps between speed reports")
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false",
                   help="the base unquantized (a CPU check, or a GPU with "
                        "room)")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--allow-template-mismatch", action="store_true")
    p.add_argument("--restart", action="store_true",
                   help="drop the checkpoints and start over")
    p.add_argument("--seed", type=int, default=0)
    # merge, gguf and modelfile are flow.train.sft's.
    for name in ("merge", "gguf", "modelfile"):
        sub.add_parser(name, help=f"flow.train.sft {name}, on the DPO run",
                       add_help=False)
    a, rest = ap.parse_known_args(argv)
    if a.cmd in ("merge", "gguf", "modelfile"):
        a.rest = [a.cmd] + rest
    elif rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    return a


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("merge", "gguf", "modelfile"):
        return sft.main(argv)
    o = parse(argv)
    if o.cmd == "stats":
        stats(o)
    else:
        train(o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
