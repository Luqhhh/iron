"""One hash-pinned zero-fit probe, with independent Decimal package replay."""
import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['sample_id', 'pred_tap_iron', 'pred_tap_time_len']
ZIP_NAME = 'Luqhhh_bf_tap_predict_round2.zip'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def rows(payload, ids):
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != COLUMNS:
        raise ValueError('Official result columns required')
    result = list(reader)
    if len(result) != 322 or [r['sample_id'] for r in result] != ids or len(set(ids)) != 322:
        raise ValueError('322 unique IDs in official template order required')
    for r in result:
        for name in COLUMNS[1:]:
            value = float(r[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Nonnegative finite predictions required; no clipping')
    return result


def load_zip(path, expected, ids):
    if sha(path) != expected:
        raise ValueError('Pinned parent ZIP changed')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['result.csv'] or archive.testzip() is not None:
            raise ValueError('Single-member valid-CRC archive required')
        return rows(archive.read('result.csv'), ids)


def context(spec_path):
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python 3.12 required')
    spec = json.loads(Path(spec_path).read_text())
    template = ROOT/spec['template']['path']
    if sha(template) != spec['template']['sha256']:
        raise ValueError('Pinned official template changed')
    ids = [r['sample_id'] for r in csv.DictReader(io.StringIO(template.read_text(encoding='utf-8-sig')))]
    sources = {name: load_zip(ROOT/v['path'], v['sha256'], ids) for name, v in spec['sources'].items()}
    return spec, ids, sources


def cold(spec_path, out, expected_warm):
    spec, ids, sources = context(spec_path)
    out = Path(out)
    if sha(out/'warm.json') != expected_warm:
        raise ValueError('Externally held warm receipt changed')
    warm = json.loads((out/'warm.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    if os.getpid() == warm['pid'] or warm['manifest_sha256'] != sha(out/'manifest.json'):
        raise ValueError('Independent process with bound manifest required')
    for path, expected in manifest['files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen release source changed')
    actual = load_zip(out/ZIP_NAME, warm['zip_sha256'], ids)
    payload = (out/'result.csv').read_bytes()
    with zipfile.ZipFile(out/ZIP_NAME) as archive:
        if payload != archive.read('result.csv'):
            raise ValueError('CSV/ZIP payload differs')
    differences = []
    with localcontext() as ctx:
        ctx.prec = 50
        for q75, a35, a60, candidate in zip(sources['q75'], sources['a35'], sources['a60'], actual):
            if candidate['pred_tap_iron'] != q75['pred_tap_iron']:
                raise ValueError('Parent iron field string changed')
            a, b = Decimal(a35['pred_tap_time_len']), Decimal(a60['pred_tap_time_len'])
            # Independent endpoint reconstruction, rather than the warm formula.
            v36 = (Decimal('.6')*a-Decimal('.35')*b)/Decimal('.25')
            n0048 = (Decimal('.65')*b-Decimal('.4')*a)/Decimal('.25')
            expected = Decimal(q75['pred_tap_time_len'])+Decimal('.05')*(v36-n0048)
            differences.append(float(abs(Decimal(candidate['pred_tap_time_len'])-expected)))
    if max(differences) > 1e-10:
        raise ValueError('Independent endpoint arithmetic differs')
    checked = dict(status='passed', G0='passed_independent_package_replay', pid=os.getpid(),
        rows=322, unique_ids=322, official_order=True, crc=True, unchanged_iron_string_mismatches=0,
        maximum_decimal_arithmetic_difference=max(differences), warm_sha256=expected_warm,
        zip_sha256=sha(out/ZIP_NAME), csv_sha256=sha(out/'result.csv'),
        new_fits=0, training_data_reads=0, platform_score=None, uploads=0, desktop_writes=0)
    write_new(out/'package-audit.json', checked)


def build(spec_path, out):
    spec, ids, sources = context(spec_path)
    if (spec['candidate'] != 'EMA_Q75_N_TO_V36_P05' or spec['source_candidate'] != 'N_TO_V36005'
            or spec['weight_transfer'] != .05 or spec['budget']['new_fits'] != 0):
        raise ValueError('Frozen single-probe scope required')
    out = Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'local/runs'):
        raise ValueError('Fresh private release required')
    reviewed = ROOT/spec['review']['path']
    if sha(reviewed) != spec['review']['sha256']:
        raise ValueError('Bound combination review changed')
    report = json.loads(reviewed.read_text())
    if report['selected_on_development'] != spec['source_candidate'] or not report['selected_four_seed_gate']:
        raise ValueError('Frozen review selected a different or unstable probe')
    independent = ROOT/spec['review_audit']['path']
    if sha(independent) != spec['review_audit']['sha256'] or json.loads(independent.read_text())['status'] != 'passed':
        raise ValueError('Independent local scoring audit required')
    files = {str(Path(spec_path).resolve()): sha(spec_path), str(Path(__file__).resolve()): sha(__file__),
        str(ROOT/spec['protocol']): sha(ROOT/spec['protocol']), str(reviewed): sha(reviewed),
        str(independent): sha(independent), str(ROOT/spec['template']['path']): spec['template']['sha256']}
    files.update({str(ROOT/v['path']): v['sha256'] for v in spec['sources'].values()})
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'manifest.json', dict(spec=spec, files=files,
        authorization='user_requested_plan_implementation_and_standing_optimization_authorization',
        inference='deterministic_recombination_of_previously_audited_delivery_fields'))
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n'); writer.writerow(COLUMNS)
    for q75, a35, a60 in zip(sources['q75'], sources['a35'], sources['a60']):
        value = float(q75['pred_tap_time_len'])+.2*(float(a35['pred_tap_time_len'])-float(a60['pred_tap_time_len']))
        writer.writerow([q75['sample_id'], q75['pred_tap_iron'], repr(value)])
    payload = stream.getvalue().encode('utf-8'); rows(payload, ids)
    with (out/'result.csv').open('xb') as target:
        target.write(payload)
    with zipfile.ZipFile(out/ZIP_NAME, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('result.csv', payload)
    write_new(out/'warm.json', dict(pid=os.getpid(), manifest_sha256=sha(out/'manifest.json'),
        zip_sha256=sha(out/ZIP_NAME), csv_sha256=sha(out/'result.csv')))
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--cold', '--spec', str(spec_path),
        '--output', str(out), '--warm-sha', sha(out/'warm.json')], check=True)
    print(json.dumps(dict(candidate=spec['candidate'], G0='passed', zip_sha256=sha(out/ZIP_NAME), package=str(out/ZIP_NAME))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default=ROOT/'configs/q75_combination_review/RELEASE.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cold', action='store_true')
    parser.add_argument('--warm-sha')
    args = parser.parse_args()
    if args.cold:
        cold(args.spec, args.output, args.warm_sha)
    else:
        build(args.spec, args.output)


if __name__ == '__main__':
    main()
