import csv
import io

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.de3_user_release import DE3Iron, release_payload
from bf_tap_r2.submission import validate_result


class Member:
    def __init__(self, iron, time):
        self.iron, self.time = iron, time

    def predict(self, frame):
        return np.tile([self.iron, self.time], (len(frame), 1))


def test_replaces_existing_component_and_preserves_time_strings():
    ids = [f"R2S2_TEST_{i:012X}" for i in range(322)]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["sample_id", "pred_tap_iron", "pred_tap_time_len"])
    writer.writerows((sid, "150", "001.230000e+02") for sid in ids)
    parent = stream.getvalue().encode()
    result = validate_result(release_payload(parent, ids, np.full(322, 200.), np.full(322, 230.)), ids)
    assert {r["pred_tap_iron"] for r in result} == {"165"}
    assert {r["pred_tap_time_len"] for r in result} == {"001.230000e+02"}
    # Blending the whole current package at .5 would erroneously produce 190.


def test_ensemble_averages_only_the_iron_output_and_rejects_labels():
    model = DE3Iron([Member(10, 1000), Member(20, -1000), Member(30, 700)])
    np.testing.assert_array_equal(model.predict(pd.DataFrame({"sample_id": ["x", "y"]})), [20, 20])
    with pytest.raises(ValueError, match="labels"):
        model.predict(pd.DataFrame({"tap_time_len": [1]}))
    with pytest.raises(ValueError, match="three"):
        DE3Iron([Member(1, 2)])


def test_component_replacement_checks_shape_and_clips_only_changed_target():
    ids = [f"R2S2_TEST_{i:012X}" for i in range(322)]
    parent = ("sample_id,pred_tap_iron,pred_tap_time_len\n" +
              "".join(f"{sid},1,9.000\n" for sid in ids)).encode()
    with pytest.raises(ValueError, match="shape"):
        release_payload(parent, ids, np.ones(321), np.ones(322))
    rows = validate_result(release_payload(parent, ids, np.full(322, 10.), np.zeros(322)), ids)
    assert all(r["pred_tap_iron"] == "0" and r["pred_tap_time_len"] == "9.000" for r in rows)
