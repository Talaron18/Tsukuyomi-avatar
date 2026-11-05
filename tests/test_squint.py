"""
flow.squint: the eye smile that shows only on a closing eye. The detection
runs on a synthetic Core stand-in (no moc, no GL), the coupling on dicts of
parameters, and the stream on the test rig with a stub layer writing the
smile. The sample rigs are checked only when the Cubism Core and the sample
models are on this machine.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from flow import squint as sq
from flow import stream as fs
from l2dprobe import core
from tests.test_live import FPS, _rig


class FakeCore:
    """One eye of two drawables. The lid (drawable 0) closes with the open
    parameter; the smile arcs it by `smile_gain * smile * (1 - open_frac)`
    (hidden at the open eye, the usual Cubism keyform grid) plus a
    `flat * smile` that shows at any opening (a rig whose smile works open)."""

    param_ids = ["ParamEyeLOpen", "ParamEyeLSmile"]
    drawable_ids = ["lid", "ball"]
    vertex_counts = np.array([3, 3])

    def __init__(self, smile_gain=0.1, flat=0.0, smile_max=1.0, open_max=1.2):
        self.param_min = np.array([0.0, 0.0])
        self.param_max = np.array([open_max, smile_max])
        self.param_default = np.array([1.0, 0.0])
        self.gain, self.flat = smile_gain, flat
        self.v = np.zeros(2)

    offsets = property(lambda self: np.array([0, 3, 6]))

    def set(self, values=None, parts=None, reset=True):
        self.v = np.array(self.param_default)
        for k, x in (values or {}).items():
            self.v[self.param_ids.index(k)] = x

    def all_vertices(self):
        o, s = self.v
        frac = min(o, 1.0)
        lid = np.array([[0, 1.0 * frac], [1, 1.0 * frac + self.gain * s
                                          * (1 - frac) + self.flat * s],
                        [2, 0.5]])
        ball = np.array([[0, 0.0], [1, 0.0], [2, 0.0]])
        return np.concatenate([lid, ball])


def test_measure_finds_a_smile_that_only_shows_on_a_closing_eye():
    r = sq.measure(FakeCore(), "ParamEyeLSmile", "ParamEyeLOpen")
    assert r["needed"] and r["effect_rest"] == 0.0 and r["effect_part"] > 0
    assert (r["rest"], r["closed"], r["smile"]) == (1.0, 0.0, (0.0, 1.0))


def test_measure_leaves_a_smile_that_shows_open_alone():
    r = sq.measure(FakeCore(smile_gain=0.0, flat=0.1), "ParamEyeLSmile",
                   "ParamEyeLOpen")
    assert not r["needed"] and r["effect_rest"] > 0
    # mostly visible on the open eye with a little more when closing
    r = sq.measure(FakeCore(smile_gain=0.02, flat=0.1), "ParamEyeLSmile",
                   "ParamEyeLOpen")
    assert not r["needed"]
    # a smile that moves nothing at all is nothing to couple
    r = sq.measure(FakeCore(smile_gain=0.0), "ParamEyeLSmile",
                   "ParamEyeLOpen")
    assert not r["needed"]


def test_measure_without_the_parameters_or_a_range():
    m = FakeCore()
    assert sq.measure(m, "ParamEyeRSmile", "ParamEyeLOpen") is None
    assert sq.measure(FakeCore(smile_max=0.0), "ParamEyeLSmile",
                      "ParamEyeLOpen") is None


def test_detect_is_lazy_cached_and_unknown_without_a_core(monkeypatch,
                                                           tmp_path):
    loads = []

    class Stub:
        @staticmethod
        def available():
            return True

        @staticmethod
        def load(path):
            loads.append(path)
            return FakeCore()

    sq._CACHE.clear()
    monkeypatch.setattr(sq, "_core", Stub)
    path = str(tmp_path / "a.model3.json")
    got = sq.detect(path)
    assert sorted(got) == ["L"]            # the R parameters are not in it
    assert got["L"]["smile_id"] == "ParamEyeLSmile"
    sq.detect(path)
    assert len(loads) == 1
    # ids from a body file's declared variables
    ids = {"eye.L.smile": "ParamEyeLSmile", "eye.L.open": "ParamEyeLOpen"}
    assert "L" in sq.detect(path, ids)
    # no Core, no model, an unreadable moc: no coupling, no error
    monkeypatch.setattr(sq, "_core", None)
    assert sq.detect(str(tmp_path / "b.model3.json")) == {}

    class Broken(Stub):
        @staticmethod
        def load(path):
            raise core.IncompatibleRig("moc3 too new")

    monkeypatch.setattr(sq, "_core", Broken)
    assert sq.detect(str(tmp_path / "c.model3.json")) == {}
    assert sq.for_model(str(tmp_path / "c.model3.json")) is None
    sq._CACHE.clear()


def test_declared_ids_come_from_the_body_file(tmp_path):
    import json
    p = tmp_path / "x.body.json"
    p.write_text(json.dumps({"declared": {
        "eye.L.smile": ["ParamA", 1.0], "eye.L.open": ["ParamB", 1.0],
        "eye.R.smile": ["ParamC", -1.0], "head.yaw": ["ParamAngleX", 30]}}))
    assert sq.declared_ids(str(p)) == {"eye.L.smile": "ParamA",
                                       "eye.L.open": "ParamB"}
    assert sq.declared_ids(None) == {} == sq.declared_ids(str(tmp_path / "n"))


SIDES = {s: {"smile_id": f"ParamEye{s}Smile", "open_id": f"ParamEye{s}Open",
             "rest": 1.0, "closed": 0.0, "smile": (0.0, 1.0)} for s in "LR"}


def test_the_coupled_open_value():
    q = sq.Squint(SIDES)
    out = q.apply({"ParamEyeLSmile": 0.8})
    assert out["ParamEyeLOpen"] == pytest.approx(1 - 0.6 * 0.8)
    assert "ParamEyeROpen" not in out          # the R side has no smile
    out = q.apply({"ParamEyeLSmile": 0.3, "ParamEyeRSmile": 1.0})
    assert out["ParamEyeLOpen"] == pytest.approx(0.82)
    assert out["ParamEyeROpen"] == pytest.approx(0.4)   # never fully closed
    # neutral smile and other parameters: nothing changes, same object
    over = {"ParamEyeLSmile": 0.0, "ParamAngleX": 5.0}
    assert q.apply(over) is over


def test_a_blink_closes_further_and_a_wider_eye_is_squinted():
    q = sq.Squint(SIDES)
    smile = {"ParamEyeLSmile": 0.8}
    assert q.apply({**smile, "ParamEyeLOpen": 0.0})["ParamEyeLOpen"] == 0.0
    assert q.apply({**smile, "ParamEyeLOpen": 0.1})["ParamEyeLOpen"] == 0.1
    # a half-open eye (a beat's own open) is not opened by the smile
    assert q.apply({**smile, "ParamEyeLOpen": 0.3})["ParamEyeLOpen"] == 0.3
    # an eye already lower than the squint (the idle's, from dflt) stays
    d = {"ParamEyeLOpen": 0.2}
    assert "ParamEyeLOpen" not in q.apply(smile, d)
    # wide eyes (1.2) squint from the open value they have
    assert q.apply({**smile, "ParamEyeLOpen": 1.2})["ParamEyeLOpen"] \
        == pytest.approx(0.52)


def test_off_switch_and_k():
    assert not sq.Squint({}) and not sq.Squint(SIDES, 0.0)
    over = {"ParamEyeLSmile": 1.0}
    assert sq.Squint(SIDES, 0.0).apply(over) is over
    assert sq.Squint({}).apply(over) is over

    class A:
        no_smile_squint = True
    assert sq.from_args(A(), "x.model3.json") is None


def test_the_option_is_added_once():
    import argparse
    ap = argparse.ArgumentParser()
    sq.add_args(ap)
    sq.add_args(ap)
    assert ap.parse_args([]).no_smile_squint is False
    assert ap.parse_args(["--no-smile-squint"]).no_smile_squint is True


class _Smile:
    """A stub layer writing the smile (and, if asked, a blink) into the
    frame as the pose, micro and blink layers do."""

    def __init__(self, **vals):
        self.vals = vals

    def apply(self, t, over):
        return {**over, **self.vals}


def test_the_stream_squints_after_the_layers_and_only_when_asked(tmp_path):
    rig = _rig(tmp_path)
    sides = {"L": {"smile_id": "ParamEyeLSmile", "open_id": "ParamEyeLOpen",
                   "rest": 1.0, "closed": 0.0, "smile": (0.0, 1.0)}}

    def frame(squint, **vals):
        st = fs.MotionStream(rig, seed=1, autonomic=False, squint=squint,
                             log=lambda s: None)
        st.drift = _Smile(**vals)
        return st.frame()[0]

    on = frame(sq.Squint(sides), ParamEyeLSmile=0.8)
    assert on["ParamEyeLOpen"] == pytest.approx(0.52)
    off = frame(None, ParamEyeLSmile=0.8)
    assert "ParamEyeLOpen" not in off
    assert {k: v for k, v in on.items() if k != "ParamEyeLOpen"} == off
    # a closed eye (a blink) stays closed
    assert frame(sq.Squint(sides), ParamEyeLSmile=0.8,
                 ParamEyeLOpen=0.0)["ParamEyeLOpen"] == 0.0


def test_the_pose_layers_smile_gets_the_squint():
    """The coupling is applied to what the pose layer wrote: a smile beat
    eases in, the open value follows it down and back, a blink inside it
    wins, and the layer's own targets (the solver's) are untouched."""
    from flow import pose as pz
    tok = {"type": pz.POSE, "t": 0.0, "duration": 0.4, "hold": 0.5,
           "release": 0.4, "params": {"ParamEyeLSmile": 0.8}}
    lay = pz.PoseLayer(FPS, {"ParamEyeLSmile": 0.0, "ParamEyeLOpen": 1.0})
    lay.commit(tok)
    q = sq.Squint(SIDES)
    opens, smiles = [], []
    for f in range(int(1.5 * FPS)):
        over, _ = lay.apply(f, {}, None)
        smiles.append(over.get("ParamEyeLSmile", 0.0))
        o = q.apply(over, {"ParamEyeLOpen": 1.0})
        opens.append(o.get("ParamEyeLOpen", 1.0))
    assert max(smiles) == pytest.approx(0.8)         # the target as written
    assert min(opens) == pytest.approx(0.52)
    assert opens[0] == 1.0 and opens[-1] == pytest.approx(1.0, abs=0.05)
    # the open value falls and rises with the smile
    k = int(np.argmax(smiles))
    assert all(a >= b - 1e-9 for a, b in zip(opens[:k], opens[1:k + 1]))


SAMPLES = os.environ.get("CUBISM_SAMPLES", "")


@pytest.mark.skipif(not core.available() or not glob.glob(
    os.path.join(SAMPLES, "Hiyori", "*.model3.json")),
    reason="needs the Cubism Core and the sample models "
           "(CUBISM_SAMPLES=.../Samples/Resources)")
@pytest.mark.parametrize("name", ["Hiyori", "Mao", "Natori", "Haru"])
def test_sample_rigs_need_the_coupling(name):
    path = glob.glob(os.path.join(SAMPLES, name, "*.model3.json"))[0]
    sq._CACHE.clear()
    got = sq.detect(path)
    assert sorted(got) == ["L", "R"]
    for s in got.values():
        assert s["effect_rest"] <= 0.2 * s["effect_part"]
