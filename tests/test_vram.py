"""
flow.train.vram: the memory arithmetic behind the 14B plan, and what in the
trainer is not tied to the 8B (family detection, Modelfile context, merge
device, the io dtype). Pure Python: no torch.
"""

from __future__ import annotations

import pytest

from flow.train import chat, sft, vram


def test_family_and_template_for_qwen3_14b():
    for name in ("Qwen/Qwen3-14B", "Qwen/Qwen3-14B-Base", "qwen3-14b",
                 "/models/Qwen3-14B-Instruct"):
        assert chat.family_for(name) == "qwen3"
    assert sft.DEFAULT_BASE == "Qwen/Qwen3-8B"
    mf = chat.modelfile("/x/q14.gguf", "qwen3", 16384)
    assert "PARAMETER num_ctx 16384" in mf and "FROM /x/q14.gguf" in mf
    assert "<|im_end|>" in mf


def test_known_architectures_reproduce_the_published_sizes():
    got = {k: vram.params(vram.arch_of(k)) / 1e9 for k in vram.KNOWN}
    assert got["qwen3-8b"] == pytest.approx(8.2, abs=0.1)
    assert got["qwen3-14b"] == pytest.approx(14.8, abs=0.1)
    assert got["qwen3-4b"] == pytest.approx(4.0, abs=0.1)
    assert vram.lora_params(vram.arch_of("Qwen/Qwen3-8B")) / 1e6 \
        == pytest.approx(87.3, abs=0.2)
    assert vram.arch_of("Qwen/Qwen3-14B")["name"] == "qwen3-14b"
    assert vram.arch_of("google/gemma-3-12b-it") is None


def test_config_wins_over_the_table():
    cfg = {"hidden_size": 5120, "num_hidden_layers": 40,
           "num_attention_heads": 40, "num_key_value_heads": 8,
           "head_dim": 128, "intermediate_size": 17408, "vocab_size": 151936,
           "tie_word_embeddings": False}
    a = vram.arch_from_config(cfg)
    assert {k: a[k] for k in vram.KNOWN["qwen3-14b"]} \
        == vram.KNOWN["qwen3-14b"]


def test_14b_qlora_fits_24gib_at_the_r3_sequence_length():
    a = vram.arch_of("Qwen/Qwen3-14B")
    t = vram.train_gib(a, 4500)
    assert 17 < t["total"] < 20         # fp32 embedding and head
    assert vram.train_gib(a, 4500, io_fp32=False)["total"] \
        == pytest.approx(t["total"] - 2.9, abs=0.3)
    assert vram.train_gib(a, 6144)["total"] < 21
    assert vram.train_gib(a, 8192)["total"] < 22
    # the 8B of before: the same arithmetic gives its ~14 GiB
    assert 12 < vram.train_gib(vram.arch_of("Qwen/Qwen3-8B"), 8192)[
        "total"] < 15.5


def test_serving_14b_next_to_tts():
    a = vram.arch_of("Qwen/Qwen3-14B")
    assert vram.kv_gib(a, 16384) == pytest.approx(2.5, abs=0.05)
    assert vram.kv_gib(a, 16384, 1) == pytest.approx(1.25, abs=0.05)
    s = vram.serve_gib(a, 16384)
    assert 11 < s["total"] < 12.5
    assert vram.kv_gib(vram.arch_of("Qwen/Qwen3-8B"), 8192) \
        == pytest.approx(1.125, abs=0.05)          # the doc's ~1.2 GB


def test_merge_goes_to_the_cpu_when_the_card_is_too_small():
    assert vram.merge_fits(vram.arch_of("Qwen/Qwen3-8B"), 24.0)
    assert not vram.merge_fits(vram.arch_of("Qwen/Qwen3-14B"), 24.0)
    assert vram.merge_fits(vram.arch_of("Qwen/Qwen3-14B"), 48.0)


def test_io_dtype_is_bf16_only_for_big_bases_unless_asked():
    cfg8 = {"hidden_size": 4096, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "head_dim": 128, "intermediate_size": 12288,
            "vocab_size": 151936}
    cfg14 = {"hidden_size": 5120, "num_hidden_layers": 40,
             "num_attention_heads": 40, "num_key_value_heads": 8,
             "head_dim": 128, "intermediate_size": 17408,
             "vocab_size": 151936}

    class O:
        io_dtype = "auto"
    assert not sft.io_bf16(O, cfg8) and sft.io_bf16(O, cfg14)
    O.io_dtype = "bf16"
    assert sft.io_bf16(O, cfg8)
    O.io_dtype = "fp32"
    assert not sft.io_bf16(O, cfg14)


def test_report_text_names_the_parts():
    t = vram.report(vram.arch_of("Qwen/Qwen3-14B"), 5000)
    for w in ("14.8B", "GiB of 24", "--io-dtype bf16", "num_ctx 16384",
              "merge on the CPU"):
        assert w in t
