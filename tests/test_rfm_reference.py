import hashlib
import json

import pytest

from bf_tap_r2.rfm_reference import unique_events, validate_native_audits
from bf_tap_r2.rfm_reference_identity import verify_source_bytes


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_source_bridge_only_allows_git_identical_line_endings():
    lf = b'a = 1\nb = 2\n'
    crlf = lf.replace(b'\n', b'\r\n')
    assert verify_source_bytes(lf, lf, digest(lf)) == 'exact'
    assert verify_source_bytes(lf, lf, digest(crlf)) == 'CRLF_only'
    with pytest.raises(ValueError):
        verify_source_bytes(lf.replace(b'1', b'3'), lf, digest(lf))
    with pytest.raises(ValueError):
        verify_source_bytes(lf, lf, '0' * 64)


def test_unique_completed_reference_units_reject_duplicates(tmp_path):
    p = tmp_path / 'events.jsonl'
    row = dict(event='complete', seed=42, fold=0)
    p.write_text(json.dumps(row)+'\n')
    assert len(unique_events(p, lambda r:(r['seed'], r['fold']))) == 1
    p.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='Duplicate'):
        unique_events(p, lambda r:(r['seed'], r['fold']))


def test_native_audits_require_both_development_and_confirmation():
    seven, twelve = {}, {}
    for seed in [42, 3407, 7777, 12011]:
        for fold in range(5):
            a = (f'development-r1/tap_time_len-tabm_plr001-s{seed}-f{fold}.npy' if seed in [42,3407]
                 else f'confirmation-r1/seed-{seed}-fold-{fold}.npy')
            b = (f'development-r1/joint-joint_plr001-s{seed}-f{fold}.npy' if seed in [42,3407]
                 else f'confirmation-r1/seed-{seed}-fold-{fold}.npy')
            seven['local/runs/round2-v7-periodic-networks/'+a]='a'*64
            twelve['local/runs/round2-v12-joint-tabm/'+b]='b'*64
    a7=dict(status='PASS', prediction_hashes=seven)
    a12=dict(status='passed', prediction_count=20, hashes=twelve)
    validate_native_audits(a7,a12)
    a12['hashes']=dict(list(twelve.items())[:10])
    with pytest.raises(ValueError, match='complete original'):
        validate_native_audits(a7,a12)
