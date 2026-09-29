"""Continuous score bounds via exact certificates on a fixed simplex tiling."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

import yaml

from .v41_concavity_bound import continuous_bound, upper_at


def canonical_cells(denominator, dimension):
    n = denominator
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or dimension not in (1, 2):
        raise ValueError('Positive integer denominator and dimension 1/2 required')
    if dimension == 1:
        return [{'id': f'I:{i}', 'vertices': [[i], [i+1]]} for i in range(n)]
    cells = []
    for i in range(n):
        for j in range(n-i):
            cells.append({'id': f'L:{i}:{j}', 'vertices': [[i,j],[i+1,j],[i,j+1]]})
            if i+j <= n-2:
                cells.append({'id': f'U:{i}:{j}', 'vertices': [[i+1,j],[i+1,j+1],[i,j+1]]})
    return cells


def exact_coarse(design, vertices):
    record = continuous_bound(design, vertices)
    anchor = min(record['anchors'], key=lambda a: max(F(c['exact_upper'])
                 for c in record['anchors'][a]['vertex_certificates']))
    certs = record['anchors'][anchor]['vertex_certificates']
    return {'exact_upper': str(max(F(c['exact_upper']) for c in certs)),
            'anchor': anchor, 'vertices': vertices, 'vertex_certificates': certs}


def partitioned_bound(design, denominator, dimension):
    cells = canonical_cells(denominator, dimension)
    vertices = sorted({tuple(v) for c in cells for v in c['vertices']})
    cache = {(anchor, vertex): upper_at(design, anchor, [str(F(v, denominator)) for v in vertex])
             for anchor in design for vertex in vertices}
    result = []
    for cell in cells:
        choices = []
        for index, anchor in enumerate(design):
            certs = [cache[anchor,tuple(v)] for v in cell['vertices']]
            if all(c is not None for c in certs):
                choices.append((max(F(c['exact_upper']) for c in certs), index, anchor, certs))
        if not choices:
            raise ValueError(f'No common anchor certifies entire cell {cell["id"]}')
        bound, _, anchor, certs = min(choices, key=lambda v: (v[0],v[1]))
        result.append({**cell, 'anchor': anchor, 'exact_upper': str(bound),
                       'vertex_certificates': certs})
    maximum = max(F(c['exact_upper']) for c in result)
    return {'denominator': denominator, 'dimension': dimension, 'cells': result,
            'vertex_count': len(vertices), 'exact_upper': str(maximum),
            'upper': math.nextafter(float(maximum), math.inf)}


def analyze(spec, base_spec):
    old_time = exact_coarse(base_spec['time_observations'], base_spec['time_domain_vertices'])
    new_time = exact_coarse(spec['time_observations'], spec['time_domain_vertices'])
    coarse_iron = exact_coarse(spec['iron_observations'], spec['iron_domain_vertices'])
    time = partitioned_bound(spec['time_observations'], spec['grid_denominator'], 2)
    iron = partitioned_bound(spec['iron_observations'], spec['grid_denominator'], 1)
    if not F(time['exact_upper']) <= F(new_time['exact_upper']) <= F(old_time['exact_upper']):
        raise ValueError('Refinement or added observation weakened the certified time bound')
    if F(iron['exact_upper']) > F(coarse_iron['exact_upper']):
        raise ValueError('Iron refinement weakened its certified bound')
    _, score, error = spec['iron_observations']['W05']
    baseline_lower = F(str(score))-F(str(error))
    gain = F(iron['exact_upper'])-baseline_lower
    upper = max(F(0), F(time['exact_upper'])+gain)
    old_upper = F(old_time['exact_upper'])+F(coarse_iron['exact_upper'])-baseline_lower
    coarse_upper = F(new_time['exact_upper'])+F(coarse_iron['exact_upper'])-baseline_lower
    return {'time': time, 'iron': iron,
            'coarse_controls': {'old_time': old_time, 'new_time': new_time, 'iron': coarse_iron,
                                'old_family_exact_upper': str(old_upper),
                                'new_family_exact_upper': str(coarse_upper)},
            'iron_reference_exact_lower': str(baseline_lower),
            'iron_gain_exact_upper': str(gain), 'family_exact_upper': str(upper),
            'family_upper': math.nextafter(float(upper), math.inf),
            'milestones': [{'target': float(t), 'excluded_under_assumptions': upper<F(str(t)),
                            'exact_margin_upper_minus_target': str(upper-F(str(t)))}
                           for t in spec['platform_milestones']],
            'new_fits': 0, 'label_reads': 0, 'test_prediction_reads': 0,
            'packages': 0, 'desktop_writes': 0, 'uploads': 0,
            'scope': 'fixed released endpoint nonnegative convex family only; not an attainable forecast',
            'conditional_on': 'correct user-reported scores/package identities, rounding intervals and fixed-row additive WMAPE'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, output):
    root = Path(root).resolve();out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v46'):
        raise ValueError('Private V46 evidence directory required')
    spec_path = root/'configs/round2_v46/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    base_path = root/spec['base_bound_spec']
    if file_hash(base_path) != spec['base_bound_spec_sha256']:
        raise ValueError('Original V41 specification changed')
    base = yaml.safe_load(base_path.read_text())
    record = json.loads(subprocess.check_output(['git','show',
        f"{spec['new_feedback_commit']}:{spec['new_feedback_path']}"],cwd=root))[spec['new_feedback_key']]
    if (record['score'] != 96.364 or record['zip_sha256'] != spec['new_feedback_zip_sha256']
            or record['coordinates'] != [0.,.45,.55]):
        raise ValueError('H1 score/package identity changed')
    retained = {k:v for k,v in spec['time_observations'].items() if k!='H1'}
    if (retained!=base['time_observations'] or spec['iron_observations']!=base['iron_observations']
            or spec['time_domain_vertices']!=base['time_domain_vertices']
            or spec['iron_domain_vertices']!=base['iron_domain_vertices']
            or spec['time_observations']['H1']!=[['0','0.45'],'96.364','0.0005']):
        raise ValueError('Frozen observations, domain or report precision changed')
    paths = ['src/bf_tap_r2/v41_concavity_bound.py','src/bf_tap_r2/v46_family_bound.py',
             'src/bf_tap_r2/v46_bound_audit.py','configs/round2_v46/SPEC.yaml',
             spec['base_bound_spec'],spec['metric_contract'],'docs/round2_v46/PREREGISTRATION.md',
             'uv.lock','pyproject.toml']
    report = analyze(spec,base)
    report.update(source_hashes={p:file_hash(root/p) for p in paths},
                  runtime={p:importlib.metadata.version(p) for p in ['numpy','scipy']},
                  source_feedback_record_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest())
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'family_upper':report['family_upper'],'time_upper':report['time']['upper'],
                      'iron_upper':report['iron']['upper'],'milestones':report['milestones']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='local/runs/round2-v46/bound-r1.json')
    args=parser.parse_args();run(Path.cwd(),args.output)
