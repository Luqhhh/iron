"""Conditional continuous-domain certificate from three reported platform scores."""
import csv
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'local/runs/mean3-strength-bound-20261004/review-r1'
POINTS = {
    '0': ('de3_user_requested_release_2026_09_30', '96.3749'),
    '0.75': ('de3_ema_mean3_platform_feedback_20261004', '96.3977'),
    '1': ('de3_ema_mean3_q100_platform_feedback_20261004', '96.3979'),
}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def verify(files):
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Frozen source changed: '+p)


def package_rows(path, ids):
    from bf_tap_r2.submission import validate_result
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ['result.csv'] or z.testzip() is not None:
            raise ValueError('Package structure or CRC differs')
        payload = z.read('result.csv')
    return validate_result(payload, ids)


def freeze():
    if OUT.exists():
        raise FileExistsError('Bound run already consumed')
    state = read(ROOT/'EVIDENCE_STATUS.json'); points = {}; files = {}
    for q, (key, score) in POINTS.items():
        value = state[key]
        if str(value.get('score', value.get('platform_score'))) != score:
            raise ValueError('Reported observation differs')
        package = ROOT/(value['private_run']+'/DE3_IRON_USER_REQUESTED/Luqhhh_bf_tap_predict_round2.zip'
                        if q == '0' else value['package'])
        feedback = ROOT/value['platform_feedback_record' if q == '0' else 'feedback_record']
        record = read(feedback)
        if (str(record['score']) != score or record['candidate'] != value['candidate']
                or record['zip_sha256'] != value['zip_sha256']):
            raise ValueError('Feedback score/candidate/package binding differs')
        if sha(package) != value['zip_sha256']:
            raise ValueError('Reported package hash differs')
        points[q] = dict(score=score, package=str(package), feedback=str(feedback),
                         zip_sha256=value['zip_sha256'], state_key=key)
        files[str(package)] = value['zip_sha256']; files[str(feedback)] = sha(feedback)
    for p in [Path(__file__), ROOT/'docs/mean3_strength_bound/PREREGISTRATION.md',
              ROOT/'configs/round2_v0_2/metric_contract.yaml', ROOT/'src/bf_tap_r2/submission.py',
              ROOT/'复赛_test/result_template.csv', ROOT/'uv.lock', ROOT/'pyproject.toml']:
        files[str(p)] = sha(p)
    verify(files); OUT.mkdir(parents=True, exist_ok=False)
    write(OUT/'manifest.json', dict(files=files, points=points, domain=['0', '1.5'],
        score_half_width='0.00005', target='96.45', created_ns=time.time_ns(),
        source='user_reported_not_independently_verified', new_fits=0, label_reads=0, packages=0))
    print(json.dumps(dict(status='frozen', files=len(files))))


def bound():
    m = read(OUT/'manifest.json'); verify(m['files'])
    ids = [r['sample_id'] for r in csv.DictReader((ROOT/'复赛_test/result_template.csv').open())]
    rows = {q: package_rows(v['package'], ids) for q, v in m['points'].items()}
    if len(ids) != 322 or len(set(ids)) != 322:
        raise ValueError('322 unique official IDs required')
    for values in rows.values():
        if any(a['pred_tap_iron'] != b['pred_tap_iron'] for a, b in zip(rows['0'], values)):
            raise ValueError('Unchanged iron must match original CSV strings')
    p = {q: [Fraction(r['pred_tap_time_len']) for r in v] for q, v in rows.items()}
    h = Fraction(m['score_half_width']); scores = {q: Fraction(v['score']) for q, v in m['points'].items()}
    lower = max(sum(p[q])/(1+2*(1-(scores[q]-h)/100)) for q in p)
    error = sum(abs(y-(x+Fraction(3, 4)*(z-x))) for x, y, z in zip(p['0'], p['0.75'], p['1']))
    epsilon = 50*error/lower
    intervals = {q: [scores[q]-h, scores[q]+h] for q in p}
    intervals['0.75'][0] -= epsilon; intervals['0.75'][1] += epsilon
    l0, _ = intervals['0']; la, ua = intervals['0.75']; lb, ub = intervals['1']
    if not lb > ua or min(pair[0] for pair in intervals.values()) <= 0:
        raise ValueError('Positive right-secant certificate precondition fails')
    uppers = [ua, ua+(ua-l0)/3, ub+2*(ub-la)]
    total = max(Fraction(0), *uppers)
    valid_domain = all(min(x, x+Fraction(3, 2)*(z-x)) >= 0 for x, z in zip(p['0'], p['1']))
    if not valid_domain:
        raise ValueError('Proposed domain contains negative predictions')
    with localcontext() as ctx:
        ctx.prec = 100
        outward = (Decimal(total.numerator)/Decimal(total.denominator)).quantize(Decimal('0.000000001'), rounding=ROUND_CEILING)
    result = dict(status='conditional_certificate_complete', denominator_lower=str(lower),
        q75_column_l1_error=str(error), q75_score_interval_expansion=str(epsilon),
        intervals={q: [str(x) for x in pair] for q, pair in intervals.items()},
        segment_uppers=[str(x) for x in uppers], global_upper=str(total), upward_display=str(outward),
        domain=m['domain'], target_excluded=total < Fraction(m['target']),
        excludes_96_4=total < Fraction('96.4'), positive_right_secant=True,
        iron_string_mismatches=0, valid_nonnegative_domain=True, rows=322,
        manifest_sha256=sha(OUT/'manifest.json'), label_reads=0, new_fits=0, new_packages=0,
        not_a_measured_or_attainable_score=True)
    write(OUT/'certificate.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'upward_display', 'target_excluded', 'excludes_96_4')}))


def audit():
    m = read(OUT/'manifest.json'); result = read(OUT/'certificate.json'); verify(m['files'])
    with localcontext() as ctx:
        ctx.prec = 100
        points = {}; iron = None; template = [r['sample_id'] for r in csv.DictReader((ROOT/'复赛_test/result_template.csv').open())]
        for q, value in m['points'].items():
            record = read(value['feedback'])
            assert str(record['score']) == value['score'] and record['zip_sha256'] == value['zip_sha256']
            with zipfile.ZipFile(value['package']) as z:
                assert z.namelist() == ['result.csv'] and z.testzip() is None
                rows = list(csv.DictReader(io.StringIO(z.read('result.csv').decode())))
            assert [r['sample_id'] for r in rows] == template and len(rows) == 322 and len(set(template)) == 322
            col = [r['pred_tap_iron'] for r in rows]
            if iron is None:
                iron = col
            assert col == iron
            points[q] = [Decimal(r['pred_tap_time_len']) for r in rows]
            assert all(v.is_finite() and v >= 0 for v in points[q])
        lo = {q: Decimal(v['score'])-Decimal('0.00005') for q, v in m['points'].items()}
        hi = {q: Decimal(v['score'])+Decimal('0.00005') for q, v in m['points'].items()}
        denom = max(sum(values)/(Decimal(3)-lo[q]/50) for q, values in points.items())
        residual = sum(abs(y-Decimal('.25')*x-Decimal('.75')*z)
                       for x, y, z in zip(points['0'], points['0.75'], points['1']))
        eps = Decimal(50)*residual/denom
        lo['0.75'] -= eps; hi['0.75'] += eps
        assert lo['1'] > hi['0.75'] and min(lo.values()) > 0
        cells = [hi['0.75'], (4*hi['0.75']-lo['0'])/3, 3*hi['1']-2*lo['0.75']]
        maximum = max(Decimal(0), *cells)
        def decode(value):
            numerator, _, denominator = value.partition('/')
            return Decimal(numerator)/(Decimal(denominator) if denominator else 1)
        pairs = [(denom, result['denominator_lower']), (residual, result['q75_column_l1_error']),
                 (eps, result['q75_score_interval_expansion']), (maximum, result['global_upper'])]
        pairs += list(zip(cells, result['segment_uppers']))
        assert all(abs(x-decode(y)) < Decimal('1e-80') for x, y in pairs)
        assert maximum.quantize(Decimal('.000000001'), rounding=ROUND_CEILING) == Decimal(result['upward_display'])
        assert result['target_excluded'] == (maximum < Decimal(m['target']))
        assert result['excludes_96_4'] == (maximum < Decimal('96.4'))
        assert m['domain'] == ['0', '1.5'] and result['domain'] == m['domain']
        assert all(Decimal('1.5')*z-Decimal('.5')*x >= 0 for x, z in zip(points['0'], points['1']))
        assert result['manifest_sha256'] == sha(OUT/'manifest.json')
    write(OUT/'independent-audit.json', dict(status='passed', arithmetic='Decimal_100_from_original_CSV',
        continuous_segments=3, certificate_sha256=sha(OUT/'certificate.json'),
        manifest_sha256=sha(OUT/'manifest.json'), new_fits=0, label_reads=0, new_packages=0))
    print(json.dumps(dict(status='passed', upward_display=result['upward_display'])))


if __name__ == '__main__':
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python3.12 required')
    {'freeze': freeze, 'bound': bound, 'audit': audit}[sys.argv[1]]()
