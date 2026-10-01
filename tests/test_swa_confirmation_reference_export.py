"""Reference admission rejects duplicate coverage, populations and changed bytes."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_swa_confirmation_references.py'
spec = importlib.util.spec_from_file_location('swa_reference_export', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def parts():
    ids = np.array([f'sample-{i:04}' for i in range(2754)])
    return [{'ids': ids[indices], 'q75': indices.astype(float) + 1}
            for indices in np.array_split(np.arange(2754)[::-1], 5)]


def test_original_query_order_is_explicitly_reindexed_without_cross_seed_average():
    ids, values, folds = exporter.close_seed(parts())
    np.testing.assert_array_equal(values, np.arange(2754) + 1)
    native_order = ids[::-1]
    _, second_values, _ = exporter.close_seed(parts(), native_order)
    np.testing.assert_array_equal(second_values, values[::-1])
    assert set(folds) == {0, 1, 2, 3, 4}


def test_duplicate_id_cannot_masquerade_as_complete_coverage():
    duplicate = parts()
    duplicate[1]['ids'][0] = duplicate[0]['ids'][0]
    with pytest.raises(ValueError, match='duplicated'):
        exporter.close_seed(duplicate)


def test_different_population_cannot_be_imported():
    ids, _, _ = exporter.close_seed(parts())
    ids[0] = 'different-sample'
    with pytest.raises(ValueError, match='populations'):
        exporter.close_seed(parts(), ids)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1.])
def test_invalid_prediction_has_no_export_or_clipping(invalid):
    incomplete = parts()
    incomplete[0]['q75'][0] = invalid
    with pytest.raises(ValueError, match='Invalid Q75'):
        exporter.close_seed(incomplete)


def test_changed_frozen_evidence_is_rejected(tmp_path):
    evidence = tmp_path / 'original.json'
    evidence.write_text('original')
    frozen = {'original.json': exporter.sha(evidence)}
    evidence.write_text('changed')
    with pytest.raises(ValueError, match='Frozen file changed'):
        exporter.verify_files(tmp_path, frozen)
