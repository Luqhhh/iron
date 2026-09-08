import json

import pytest

from bf_tap.artifacts import atomic_write_json, file_sha256, stable_digest
from bf_tap.exceptions import ProtectedLabelError
from bf_tap.optimization.lifecycle import authorize_protected_operation
from bf_tap.protection import load_protection_policy


def test_synthetic_authorization_fail_closed_and_append_only(tmp_path):
    policy = load_protection_policy("configs/protection.yaml")
    manifest = tmp_path / "synthetic_manifest.json"
    algorithm = {"model": "synthetic_no_real_labels"}
    atomic_write_json(manifest, {"schema_version": "frozen-algorithm-v3", "algorithm_identity": "synthetic",
                                "policy_digest": policy.digest, "algorithm": algorithm,
                                "algorithm_sha256": stable_digest(algorithm)})
    authorization = {"schema_version": "protected-authorization-v1", "lifecycle": "holdout_scoring",
                     "policy_digest": policy.digest, "frozen_manifest_sha256": file_sha256(manifest),
                     "approval_id": "synthetic-holdout"}
    kwargs = dict(policy=policy, lifecycle="holdout_scoring", purpose="frozen_report", frozen_manifest=manifest,
                  ledger=tmp_path / "synthetic_ledger.jsonl", algorithm_identity="synthetic")
    for auth in (None, {**authorization, "frozen_manifest_sha256": "0" * 64},
                 {**authorization, "lifecycle": "final_training"}):
        with pytest.raises(ProtectedLabelError):
            authorize_protected_operation(**kwargs, authorization=auth)
    for purpose in ("model_selection", "feature_cache", "development"):
        with pytest.raises(ProtectedLabelError):
            authorize_protected_operation(**{**kwargs, "purpose": purpose}, authorization=authorization)
    assert not kwargs["ledger"].exists()
    authorize_protected_operation(**kwargs, authorization=authorization)
    before = kwargs["ledger"].read_bytes()
    with pytest.raises(ProtectedLabelError):
        authorize_protected_operation(**kwargs, authorization=authorization)
    assert kwargs["ledger"].read_bytes() == before
    with pytest.raises(ProtectedLabelError):
        authorize_protected_operation(**{**kwargs, "lifecycle": "final_training", "purpose": "frozen_final_fit"}, authorization=authorization)
    authorize_protected_operation(**{**kwargs, "lifecycle": "final_training", "purpose": "frozen_final_fit"},
                                  authorization={**authorization, "lifecycle": "final_training", "approval_id": "synthetic-final"})
    assert kwargs["ledger"].read_bytes().startswith(before)
    assert len([json.loads(line) for line in kwargs["ledger"].read_text().splitlines()]) == 2
