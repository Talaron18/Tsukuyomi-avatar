"""
Memory arithmetic for the bases the trainer and the live server are planned
for, in plain Python (no torch): how much of a 24 GB card a QLoRA run, a merge
and an Ollama server take. docs/embodiment.md ("Memory (estimates, 24 GB)",
"A 14B student") shows the numbers; `flow.train.sft stats` prints this for the
run's base and sequence length; `merge` uses it to pick the device.

The architectures are the models' published configs (Qwen3 model cards:
hidden size, layers, query / key-value heads, head dim, FFN width, vocabulary,
tied embeddings). They were written down from the cards without being able to
fetch config.json in the session that added this file, so a config.json read
from the base itself (arch_from_config) always wins; check KNOWN against it
the first time a base is used.

What the estimate counts, per GiB (2**30 bytes):
  weights   4-bit NF4 blocks with double quantisation (~0.52 bytes a weight
            of the transformer layers), the embedding and the output head in
            16 bit, or in 32 bit when peft's prepare_model_for_kbit_training
            has run (it casts every non-quantised 16-bit parameter to fp32;
            `io_fp32`): the head and the embedding are 2 x vocab x hidden,
            which is what makes a 14B hard on 24 GB (3.1 GB in bf16, 6.2 GB
            in fp32).
  lora      rank r on q, k, v, o, gate, up, down: fp32 weights + fp32 grads +
            two 8-bit Adam states.
  acts      gradient checkpointing keeps each layer's input (seq x hidden x
            2 bytes) and recomputes one layer at a time: its FFN
            intermediates (3 x seq x ffn x 2 bytes) and attention buffers
            (sdpa) at peak, plus a flat 1 GiB for the allocator and kernels.
  logits    fp32 logits, their softmax and gradient at the labelled
            positions only (flow.train.sft keeps logits_to_keep): 3 x
            labelled x vocab x 4 bytes.
  kv        the KV cache of a served model: 2 x layers x kv heads x head dim
            x 2 bytes (f16) a token.
"""

from __future__ import annotations

import re

GIB = 2 ** 30
KNOWN = {
    "qwen3-4b": dict(hidden=2560, layers=36, heads=32, kv_heads=8,
                     head_dim=128, ffn=9728, vocab=151936, tied=True),
    "qwen3-8b": dict(hidden=4096, layers=36, heads=32, kv_heads=8,
                     head_dim=128, ffn=12288, vocab=151936, tied=False),
    "qwen3-14b": dict(hidden=5120, layers=40, heads=40, kv_heads=8,
                      head_dim=128, ffn=17408, vocab=151936, tied=False),
}
# Q4_K_M file sizes in GiB as published for these bases (llama.cpp quants of
# the Qwen3 releases): used for the server estimate, not for training.
Q4_K_M_GIB = {"qwen3-4b": 2.4, "qwen3-8b": 4.7, "qwen3-14b": 8.4}


def key_of(base):
    m = re.search(r"qwen-?3[-_ ]?(\d+(?:\.\d+)?)b", (base or "").lower())
    return f"qwen3-{m.group(1)}b" if m else None


def arch_of(base):
    """The architecture of a base by its name (KNOWN), or None."""
    k = key_of(base)
    return dict(KNOWN[k], name=k) if k in KNOWN else None


def arch_from_config(cfg):
    """The architecture from a transformers config (an object or a dict:
    config.json), the source to trust over KNOWN."""
    def g(k, default=None):
        return cfg.get(k, default) if isinstance(cfg, dict) \
            else getattr(cfg, k, default)
    heads = int(g("num_attention_heads"))
    hidden = int(g("hidden_size"))
    return dict(hidden=hidden, layers=int(g("num_hidden_layers")),
                heads=heads, kv_heads=int(g("num_key_value_heads", heads)),
                head_dim=int(g("head_dim", hidden // heads)),
                ffn=int(g("intermediate_size")), vocab=int(g("vocab_size")),
                tied=bool(g("tie_word_embeddings", False)), name="config")


def layer_params(a) -> int:
    q = a["hidden"] * a["heads"] * a["head_dim"]
    kv = a["hidden"] * a["kv_heads"] * a["head_dim"]
    return q + 2 * kv + q + 3 * a["hidden"] * a["ffn"]


def params(a) -> int:
    """All parameters: the layers, the embedding and, untied, the head."""
    emb = a["vocab"] * a["hidden"]
    return a["layers"] * layer_params(a) + emb * (1 if a["tied"] else 2)


def lora_params(a, rank=32) -> int:
    h, hd = a["hidden"], a["head_dim"]
    per = (rank * (h + a["heads"] * hd) * 2          # q, o
           + rank * (h + a["kv_heads"] * hd) * 2     # k, v
           + rank * (h + a["ffn"]) * 3)              # gate, up, down
    return a["layers"] * per


def train_gib(a, seq, rank=32, load_4bit=True, io_fp32=True, labelled=1000,
              ) -> dict:
    """GiB of a QLoRA step at `seq` tokens (module docstring), by part."""
    emb = a["vocab"] * a["hidden"]
    io_bytes = 4 if io_fp32 else 2
    io = emb * (1 if a["tied"] else 2) * io_bytes
    layers = a["layers"] * layer_params(a)
    w = layers * (0.52 if load_4bit else 2.0) + io
    lora = lora_params(a, rank) * (4 + 4 + 2)
    acts = (a["layers"] * seq * a["hidden"] * 2
            + 3 * seq * a["ffn"] * 2 + 4 * seq * a["hidden"] * 2) + GIB
    logits = 3 * labelled * a["vocab"] * 4
    parts = {"weights": w, "lora+optimizer": lora, "activations": acts,
             "logits": logits}
    out = {k: v / GIB for k, v in parts.items()}
    out["total"] = sum(out.values())
    return out


def kv_gib(a, ctx, bytes_per=2) -> float:
    return 2 * a["layers"] * a["kv_heads"] * a["head_dim"] * bytes_per \
        * ctx / GIB


def serve_gib(a, ctx, q4_gib=None, kv_bytes=2, overhead=0.6) -> dict:
    """GiB of an Ollama server: the Q4_K_M file, the KV cache at num_ctx and
    the compute buffers."""
    q4 = q4_gib if q4_gib is not None else Q4_K_M_GIB.get(
        a.get("name", ""), params(a) * 0.6 / GIB)
    kv = kv_gib(a, ctx, kv_bytes)
    return {"model": q4, "kv": kv, "buffers": overhead,
            "total": q4 + kv + overhead}


def merge_fits(a, gpu_gib, dtype_bytes=2, margin=0.85) -> bool:
    """Whether the unquantised model fits the card for a merge."""
    return params(a) * dtype_bytes / GIB <= gpu_gib * margin


def report(a, seq, ctx=16384, rank=32, gpu_gib=24.0) -> str:
    """The text `flow.train.sft stats` prints for a known base."""
    t = train_gib(a, seq, rank, io_fp32=True)
    t2 = train_gib(a, seq, rank, io_fp32=False)
    s = serve_gib(a, ctx)
    mp = merge_fits(a, gpu_gib)
    return "\n".join([
        f"memory ({a.get('name', '?')}, {params(a) / 1e9:.1f}B parameters, "
        f"QLoRA r{rank}, {seq} tokens, batch 1):",
        "  " + ", ".join(f"{k} {v:.1f}" for k, v in t.items() if k != "total")
        + f" -> {t['total']:.1f} GiB of {gpu_gib:g} (embedding and head fp32, "
        "as prepare_model_for_kbit_training leaves them)",
        f"  with the embedding and the head kept bf16 (--io-dtype bf16): "
        f"{t2['total']:.1f} GiB",
        f"  Ollama Q4_K_M at num_ctx {ctx}: {s['model']:.1f} + KV "
        f"{s['kv']:.1f} + buffers {s['buffers']:.1f} = {s['total']:.1f} GiB",
        "  merge in fp16 on the card: "
        + ("fits" if mp else "does not fit: merge on the CPU")])
