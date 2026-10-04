from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.iron_strength_correction import wmape
from bf_tap_r2.slot_screen import read_json


def test_wmape_hand_value():
    assert wmape(np.array([10.0, 10.0]), np.array([9.0, 12.0])) == pytest.approx(0.15)


def test_correction_spec_documents_the_base_error():
    spec = read_json(Path("configs/iron_strength_correction/SPEC.json"))
    assert spec["new_fits"] == 0 and spec["new_packages"] == 0
    assert spec["agent_uploads"] == 0 and spec["desktop_writes"] == 0
    assert "raw V12 joint member" in spec["correction"]
    assert spec["withdrawn_candidate"].startswith("DE3_IRON_STRENGTH_100")
    assert 0.5 in spec["amplification_grid"] and 1.0 in spec["amplification_grid"]


def test_correction_report_reproduces_the_withdrawal():
    report = Path("local/runs/iron-strength-correction-20261005/review-r1/report.json")
    if not report.exists():
        pytest.skip("private correction evidence not present")
    result = read_json(report)["result"]
    assert result["released_plain_mean3_is_worse_on_both_splits"] is True
    assert result["q1_is_worse_or_flat_on_both_splits"] is True
    assert result["released_plain_mean3_gain_mean"] < -0.015
    for split in ("42", "3407"):
        curve = result["splits"][split]["score_gain_vs_incumbent"]
        assert abs(curve["0.5"]) < 1e-12          # the incumbent is exactly the q=0.5 point
        assert curve["1.0"] < 0                    # raising q loses
        assert curve["0.0"] < curve["0.25"] < 0    # lowering q also loses
    assert result["splits"]["42"]["optimum_q"] in (0.5, 0.75)


def test_withdrawn_package_is_marked_and_status_matches():
    marker = Path("local/runs/iron-strength-release-20261005/release-r1/WITHDRAWN.json")
    if not marker.exists():
        pytest.skip("private withdrawal marker not present")
    saved = read_json(marker)
    assert saved["status"] == "withdrawn_must_not_be_uploaded"
    assert saved["deleted_evidence"] is False and saved["package_retained"] is True
    assert saved["agent_uploads"] == 0 and saved["desktop_writes"] == 0
    assert saved["measured_relative_to_incumbent"]["mean"] < -0.015
