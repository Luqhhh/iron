import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES, SUBMISSION_COLUMNS
from bf_tap_r2.v2_release import load_v2, payload_v2
from bf_tap_r2.submission import validate_result


def fixture(folder):
    ids = [f"R2S2_TEST_{i:012X}" for i in range(322)]
    pd.DataFrame({"sample_id": ids, "spout_no": 1}).to_csv(folder / "test_samples.csv", index=False)
    pd.DataFrame({"sample_id": ids, **{f: np.arange(322) for f in FEATURES}}).iloc[::-1].to_csv(folder / "test_features.csv", index=False)
    pd.DataFrame({"sample_id": ids, **{c: np.nan for c in SUBMISSION_COLUMNS[1:]}}).to_csv(folder / "result_template.csv", index=False)
    return ids


def test_v2_alignment_and_release(tmp_path):
    ids = fixture(tmp_path)
    test = load_v2(tmp_path, "test", 322)
    assert test.sample_id.tolist() == ids
    assert test[FEATURES[0]].tolist() == list(range(322))
    model = {"candidate": "R2_M0_GLOBAL_MEDIAN", "medians": {"tap_iron": 3.25, "tap_time_len": 7.5}}
    rows = validate_result(payload_v2(model, ids), ids)
    assert float(rows[-1]["pred_tap_iron"]) == 3.25


@pytest.mark.parametrize("damage", ["old", "duplicate", "missing", "foreign_feature", "template", "nonfinite"])
def test_v2_rejects_damaged_inputs(tmp_path, damage):
    fixture(tmp_path)
    name = "result_template" if damage == "template" else "test_features"
    p = tmp_path / f"{name}.csv"
    frame = pd.read_csv(p)
    if damage == "old":
        frame["sample_id"] = frame.sample_id.str.replace("R2S2_", "R2S_", regex=False)
    elif damage == "duplicate":
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    elif damage == "missing":
        frame = frame.iloc[:-1]
    elif damage in ("foreign_feature", "template"):
        frame.loc[0, "sample_id"] = "R2S2_TEST_FFFFFFFFFFFF"
    else:
        frame[FEATURES[0]] = frame[FEATURES[0]].astype(float)
        frame.loc[0, FEATURES[0]] = np.inf
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_v2(tmp_path, "test", 322)


def test_v2_serializer_rejects_old_ids():
    ids = [f"R2S_TEST_{i:012X}" for i in range(322)]
    with pytest.raises(ValueError):
        payload_v2({}, ids)
