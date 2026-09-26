from copy import deepcopy
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")

from bf_tap_r2.data import FEATURES
from bf_tap_r2.v8_attention import AttentionRegressor, FeatureAttentionNetwork, UniformAttention


def settings():
    spec = yaml.safe_load(Path("configs/round2_v8/SPEC.yaml").read_text())["training"]
    return {**spec, "width": 16, "heads": 2, "max_epochs": 3, "patience": 2}


def test_uniform_is_zero_logit_attention_and_removes_qk_gradients():
    torch.manual_seed(42)
    network = FeatureAttentionNetwork("learned", settings(), 3)
    layer = network.base.backbone.blocks[0]["attention"]
    control = UniformAttention(deepcopy(layer)).eval()
    zero_logits = deepcopy(layer).eval()
    for component in [zero_logits.W_q, zero_logits.W_k]:
        torch.nn.init.zeros_(component.weight)
        if component.bias is not None:
            torch.nn.init.zeros_(component.bias)
    q, kv = torch.randn(5, 1, 16), torch.randn(5, 7, 16)
    torch.testing.assert_close(control(q, kv), zero_logits(q, kv), rtol=0, atol=1e-7)
    control(q, kv).square().mean().backward()
    assert control.source.W_q.weight.grad is None
    assert control.source.W_k.weight.grad is None
    assert control.source.W_v.weight.grad.abs().sum() > 0
    layer(q, kv).square().mean().backward()
    assert layer.W_q.weight.grad.abs().sum() > 0
    assert layer.W_k.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["uniform", "learned"])
def test_refit_prediction_is_deterministic_and_cold_row_independent(tmp_path, mode):
    rng = np.random.default_rng(63)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame["sample_id"] = [f"synthetic-{i:04d}" for i in range(len(frame))]
    frame["spout_no"] = np.arange(len(frame)) % 2 + 1
    y = 100 + 3 * frame.air_volume.values + rng.normal(size=len(frame))
    model = AttentionRegressor({"attention": mode}, settings()).fit(frame, y)
    again = AttentionRegressor({"attention": mode}, settings()).fit(frame, y)
    expected = model.predict(frame)
    np.testing.assert_array_equal(expected, again.predict(frame))
    np.testing.assert_allclose(expected, model.predict(frame.iloc[::-1])[::-1], rtol=0, atol=1e-4)
    np.testing.assert_allclose(expected[:1], model.predict(frame.iloc[:1]), rtol=0, atol=1e-4)
    np.testing.assert_allclose(expected, np.r_[model.predict(frame.iloc[:40]),model.predict(frame.iloc[40:])], rtol=0, atol=1e-4)
    with (tmp_path / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    frame.to_pickle(tmp_path / "query.pkl")
    code = "import pickle,pandas as pd,numpy as np; from pathlib import Path; p=Path(__import__('sys').argv[1]); m=pickle.loads((p/'model.pkl').read_bytes()); np.save(p/'cold.npy',m.predict(pd.read_pickle(p/'query.pkl')))"
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
    np.testing.assert_array_equal(expected, np.load(tmp_path / "cold.npy"))
    assert model.metadata_["fit_rows"] == len(frame)
    assert model.metadata_["inner_fit_rows"] == 80
    assert model.metadata_["selection_stopped_epoch"] <= settings()["max_epochs"]
    assert any(g["weight_decay"] == 0 for g in model.optimizer_.param_groups)
