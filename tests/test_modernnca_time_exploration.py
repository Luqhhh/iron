"""Release boundary checks: CSV strings, official order, invalid extrapolation."""
import csv
import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('modern_release', Path(__file__).parents[1] / 'scripts/modernnca_time_exploration.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def example():
    ids = [f'R2S2_TEST_{i:06d}' for i in range(322)]
    text = 'sample_id,pred_tap_iron,pred_tap_time_len\n'
    text += ''.join(f'{sid},01.230000e+03,100\n' for sid in ids)
    return text.encode(), ids


def test_untouched_iron_strings_and_time_formula():
    parent, ids = example()
    rows = list(csv.DictReader(io.StringIO(release.payload(parent, ids, [110]*322).decode())))
    assert all(r['pred_tap_iron'] == '01.230000e+03' for r in rows)
    assert all(float(r['pred_tap_time_len']) == 102 for r in rows)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_invalid_member_refused_without_clipping(bad):
    parent, ids = example()
    with pytest.raises(ValueError, match='clipping prohibited'):
        release.payload(parent, ids, [bad]*322)


def test_permuted_or_duplicate_template_ids_refused():
    parent, ids = example()
    with pytest.raises(ValueError, match='template'):
        release.payload(parent, ids[::-1], [100]*322)
    with pytest.raises(ValueError, match='template'):
        release.payload(parent, [ids[0]]*322, [100]*322)


def test_existing_evidence_is_not_overwritten(tmp_path):
    path = tmp_path / 'receipt.json'
    release.write(path, {'first': 1})
    with pytest.raises(FileExistsError):
        release.write(path, {'second': 2})
    assert release.read(path) == {'first': 1}
