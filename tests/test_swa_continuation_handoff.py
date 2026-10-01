"""Portable imports must reject changed bytes, unsafe paths and split mismatches."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_swa_continuation_handoff.py'
spec = importlib.util.spec_from_file_location('swa_handoff', SCRIPT)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def minimal_bundle(directory):
    payload = directory / 'payload.txt'
    payload.write_text('original evidence')
    manifest = directory / 'HANDOFF_MANIFEST.json'
    manifest.write_text(json.dumps({'payload_files': {'payload.txt': {
        'bytes': payload.stat().st_size, 'sha256': handoff.sha(payload)}}}))
    return handoff.sha(manifest)


def test_external_manifest_hash_cannot_be_replaced_with_a_self_claim(tmp_path):
    expected = minimal_bundle(tmp_path)
    (tmp_path / 'HANDOFF_MANIFEST.json').write_text('{}')
    with pytest.raises(ValueError, match='External handoff manifest identity'):
        handoff.verify_bundle(tmp_path, expected)


def test_payload_tampering_is_rejected_before_reference_import(tmp_path):
    expected = minimal_bundle(tmp_path)
    (tmp_path / 'payload.txt').write_text('changed evidence!')
    with pytest.raises(ValueError, match='Changed payload'):
        handoff.verify_bundle(tmp_path, expected)


def test_unlisted_files_cannot_enter_verified_payload(tmp_path):
    expected = minimal_bundle(tmp_path)
    (tmp_path / 'unexpected.npy').write_bytes(b'not admitted')
    with pytest.raises(ValueError, match='Unexpected or missing'):
        handoff.verify_bundle(tmp_path, expected)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'refs/../../escape'])
def test_unsafe_archive_path_is_rejected(tmp_path, name):
    with pytest.raises(ValueError, match='Unsafe payload path'):
        handoff.child(tmp_path, name)


def test_symlink_cannot_redirect_import_outside_bundle(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('unadmitted')
    (bundle / 'alias').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes bundle'):
        handoff.child(bundle, 'alias')


def verified():
    return {'ids': np.array(['a', 'b', 'c']), 'columns': {
        271828: {'time': np.array([1., 2., 3.]), 'folds': np.array([0, 1, 2])},
        314159: {'time': np.array([6., 5., 4.]), 'folds': np.array([2, 0, 1])}}}


def test_native_order_preserves_specific_seed_and_returns_independent_arrays():
    imported = verified()
    values, folds = handoff.reindex_reference(imported, 314159, ['c', 'a', 'b'], [1, 2, 0])
    np.testing.assert_array_equal(values, [4., 6., 5.])
    np.testing.assert_array_equal(folds, [1, 2, 0])
    values[0] = 999.
    assert imported['columns'][314159]['time'][2] == 4.


def test_native_fold_mismatch_is_rejected():
    with pytest.raises(ValueError, match='split seed/folds differ'):
        handoff.reindex_reference(verified(), 271828, ['c', 'a', 'b'], [1, 2, 0])


@pytest.mark.parametrize('ids', [['a', 'a', 'c'], ['a', 'b', 'different']])
def test_duplicate_or_different_native_population_is_rejected(ids):
    with pytest.raises(ValueError, match='row population differs'):
        handoff.reindex_reference(verified(), 271828, ids)


def test_missing_old_confirmation_seed_cannot_be_renamed():
    with pytest.raises(ValueError, match='Unregistered reference seed'):
        handoff.reindex_reference(verified(), 7777, ['a', 'b', 'c'])
