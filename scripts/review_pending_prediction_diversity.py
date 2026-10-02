"""Label-free comparison of the three already delivered, fixed candidate ZIPs."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
POOL = ('LAPLACE_FIXED_A20', 'GAUSS1_A20', 'EMA_MEAN3_FULL_Q75')
PROTOCOL = 'docs/platform_slot_review/PREREGISTRATION.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def freeze(out):
    if out.exists() or not out.is_relative_to(ROOT / 'local/runs'):
        raise ValueError('Fresh private review required')
    state=json.loads((ROOT/'EVIDENCE_STATUS.json').read_text())
    best=state['round2_current_platform_best']
    if best['candidate']!='EMA_TIME_Q75' or best['score']!=96.392:
        raise ValueError('Registered reference changed')
    release=json.loads((ROOT/'configs/q75_laplace_release/SPEC.json').read_text())
    entries={'Q75':{'path':str(ROOT/release['parent_package']),'sha256':best['zip_sha256']}}
    queue=state['round2_current_candidate_queue']['ready_for_user_platform_testing']
    for name in POOL:
        matches=[v for v in queue if v['candidate']==name]
        if len(matches)!=1:raise ValueError('Exactly one registered package per candidate required')
        item=matches[0]
        entries[name]={'path':str(ROOT/item['package']),'sha256':item['zip_sha256']}
    for item in entries.values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Registered ZIP changed')
    sources={str(ROOT/p):sha(ROOT/p) for p in (PROTOCOL,'scripts/review_pending_prediction_diversity.py',
        'configs/protection.yaml','configs/q75_laplace_release/SPEC.json')}
    out.mkdir(parents=True)
    write_new(out/'manifest.json',dict(entries=entries,source_hashes=sources,pool=list(POOL),
        frozen_ns=time.time_ns(),training_or_test_labels_read=False,new_fits=0,packages=0,
        platform_slots_user_reported=2,automatic_ranking=False,python=sys.version))


def parse_zip(path):
    with zipfile.ZipFile(path) as z:
        if z.namelist()!=['result.csv'] or z.testzip() is not None:
            raise ValueError('Single result.csv and correct CRC required')
        payload=z.read('result.csv')
    reader=csv.DictReader(io.StringIO(payload.decode('utf-8-sig'),newline=''))
    if reader.fieldnames!=['sample_id','tap_iron','tap_time_len']:
        raise ValueError('Official prediction schema required')
    rows=list(reader)
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=322 or len(set(ids))!=322:raise ValueError('322 unique prediction IDs required')
    values={k:[float(r[k]) for r in rows] for k in ('tap_iron','tap_time_len')}
    if any(not math.isfinite(v) or v<0 for a in values.values() for v in a):
        raise ValueError('Finite nonnegative predictions required')
    return ids,rows,values,hashlib.sha256(payload).hexdigest()


def review(out):
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['pool']!=list(POOL):raise ValueError('Frozen pool changed')
    for p,h in manifest['source_hashes'].items():
        if sha(p)!=h:raise ValueError('Frozen review source changed')
    with (out/'access.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time_ns=time.time_ns(),scope='four_frozen_prediction_ZIPs_only',
            manifest_sha256=sha(out/'manifest.json'),labels_read=False))+'\n')
    packages={}
    for name,item in manifest['entries'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen package changed')
        packages[name]=parse_zip(item['path'])
    ids,base_rows,base,_=packages['Q75'];deltas={};summaries={}
    for name in POOL:
        new_ids,rows,values,csv_sha=packages[name]
        if new_ids!=ids:raise ValueError('Common official ordered IDs required')
        mismatches=sum(a['tap_iron']!=b['tap_iron'] for a,b in zip(rows,base_rows))
        if mismatches:raise ValueError('Iron raw strings changed')
        delta=[a-b for a,b in zip(values['tap_time_len'],base['tap_time_len'])]
        deltas[name]=delta
        summaries[name]=dict(csv_sha256=csv_sha,iron_string_mismatches=mismatches,
            mean_abs_time_change=math.fsum(map(abs,delta))/len(ids),
            rms_time_change=math.sqrt(math.fsum(v*v for v in delta)/len(ids)),
            maximum_abs_time_change=max(map(abs,delta)),time_increased=sum(v>0 for v in delta),
            time_decreased=sum(v<0 for v in delta),time_unchanged=sum(v==0 for v in delta))
    pairs=[]
    for left,right in itertools.combinations(POOL,2):
        a,b=deltas[left],deltas[right];den=math.sqrt(math.fsum(v*v for v in a)*math.fsum(v*v for v in b))
        pairs.append(dict(left=left,right=right,mean_abs_prediction_gap=math.fsum(abs(x-y) for x,y in zip(a,b))/len(ids),
            delta_cosine=None if den==0 else math.fsum(x*y for x,y in zip(a,b))/den,
            same_nonzero_direction=sum(x*y>0 for x,y in zip(a,b))))
    write_new(out/'report.json',dict(status='passed',rows=len(ids),summaries=summaries,pairs=pairs,
        manifest_sha256=sha(out/'manifest.json'),labels_read=False,new_fits=0,new_packages=0,
        automatic_ranking=False,interpretation='Prediction changes only; neither gains nor error independence. No weight or package selection.'))
    print(json.dumps(dict(status='passed',rows=len(ids),report_sha256=sha(out/'report.json'),pairs=pairs),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['freeze','review']);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if sys.version_info[:2]!=(3,12):raise ValueError('Locked Python3.12 required')
    globals()[args.operation](args.output.resolve())
