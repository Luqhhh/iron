import json
from pathlib import Path

import pytest
import yaml

from bf_tap_r2.incumbent_reference_cache import inspect_development,safe_child,source_child,verify_hashes
from bf_tap_r2.incumbent_reference_ledger import file_hash,write_new
from bf_tap_r2.v7_periodic import digest


def fixture(tmp_path):
    source=tmp_path/'source';cache=tmp_path/'cache';source.mkdir();cache.mkdir()
    config=source/'configs/independent_ensemble_checkpoints/SPEC.yaml';config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(dict(training=dict(tap_iron=dict(random_seed=42)))))
    native=source/'configs/strong_component_regularization/SPEC.yaml';native.parent.mkdir()
    native.write_text(yaml.safe_dump(dict(mechanisms=dict(ema_beta=.99))))
    data=source/'复赛_train';data.mkdir();(data/'train_samples.csv').write_text('synthetic metadata fixture only\n');(data/'train_features.csv').write_text('synthetic metadata fixture only\n')
    m=dict(queue='DE3',seeds=[42,3407],development=None,versions={},source_hashes={str(p.relative_to(source)):file_hash(p) for p in (config,native)},
        data_hashes={str(p.relative_to(source)):file_hash(p) for p in data.iterdir()},spec_sha256=file_hash(config))
    m['identity']=digest(m);write_new(cache/'manifest.json',m)
    write_new(cache/'summary.json',dict(selected_for_confirmation=dict(tap_iron=None)))
    write_new(cache/'audit.json',dict(status='passed',queue='DE3',native_replays=20,new_fits=0,full_batch_cold_difference=0.,
        manifest_sha256=file_hash(cache/'manifest.json'),summary_sha256=file_hash(cache/'summary.json')))
    for s in (42,3407):
        for f in range(5):
            name=f'tap_iron-DE3-s{s}-f{f}';u=cache/name;u.mkdir()
            write_new(u/'metadata.json',dict(training_seeds=[42,104729,130363],aggregation='fixed_arithmetic_mean',target='tap_iron',
                queue='DE3',seed=s,fold=f,selector_fits=2,refit_fits=2,reused_native_units=1,seconds=100))
            (u/'predictions.npz').write_bytes(b'synthetic metadata fixture')
            for t in (42,104729,130363):
                sub=u/f'training-seed-{t}';sub.mkdir()
                write_new(sub/'metadata.json',dict(model=dict(traces={p:dict(updates=100,peak_rss_mib=700) for p in ('selection','refit')})))
                write_new(sub/'complete.json',dict(identity=digest(dict(manifest_identity=m['identity'],unit=f'{name}/training-seed-{t}')),
                    key=f'{name}/training-seed-{t}',hashes={'metadata.json':file_hash(sub/'metadata.json')}))
            nested={str(p.relative_to(u)):file_hash(p) for p in u.rglob('*') if p.is_file()}
            write_new(u/'nested-hashes.json',nested)
            write_new(u/'complete.json',dict(identity=digest(dict(manifest_identity=m['identity'],unit=name)),key=name,
                hashes={n:file_hash(u/n) for n in ('predictions.npz','metadata.json','nested-hashes.json')}))
    return source,cache,{n:file_hash(cache/n) for n in ('manifest.json','summary.json','audit.json')}


def test_metadata_only_freeze_has_complete_iron_tree_and_cost_witnesses(tmp_path):
    source,cache,anchors=fixture(tmp_path);report=inspect_development(source,cache,anchors)
    assert report['status']=='passed' and report['new_fits']==0
    assert len(report['cost_witnesses'])==10 and len(report['hashes'])==103
    assert report['original_native_mechanisms']==dict(ema_beta=.99)
    # No np.load is possible for the intentionally non-NPY bytes in this fixture.


@pytest.mark.parametrize('defect',['external_anchor','snapshot','nested','unit_identity','original_source'])
def test_old_evidence_and_source_mutations_rejected_before_label_loading(tmp_path,defect):
    source,cache,anchors=fixture(tmp_path)
    if defect=='external_anchor':anchors['audit.json']='0'*64
    elif defect=='snapshot':(source/'复赛_train/train_samples.csv').write_text('changed')
    elif defect=='nested':(cache/'tap_iron-DE3-s42-f0/training-seed-104729/metadata.json').write_text('{}')
    elif defect=='unit_identity':
        p=cache/'tap_iron-DE3-s42-f0/complete.json';x=json.loads(p.read_text());x['identity']='wrong';p.write_text(json.dumps(x))
    else:
        # An unapproved private evidence source has no text-byte bridge.
        m=json.loads((cache/'manifest.json').read_text());p=source/'private.json';p.write_text('{}')
        m['source_hashes']['private.json']='0'*64;m.pop('identity');m['identity']=digest(m)
        (cache/'manifest.json').write_text(json.dumps(m));anchors['manifest.json']=file_hash(cache/'manifest.json')
        a=json.loads((cache/'audit.json').read_text());a['manifest_sha256']=anchors['manifest.json'];(cache/'audit.json').write_text(json.dumps(a));anchors['audit.json']=file_hash(cache/'audit.json')
    with pytest.raises((ValueError,AssertionError)):inspect_development(source,cache,anchors)


def test_hash_paths_cannot_escape_evidence_root(tmp_path):
    p=tmp_path/'outside';p.write_text('secret');root=tmp_path/'cache';root.mkdir()
    with pytest.raises(ValueError,match='escapes'):safe_child(root,'../outside')
    with pytest.raises(ValueError,match='escapes'):verify_hashes(root,{str(p):file_hash(p)})


def test_shared_private_runs_are_bounded_without_allowing_other_symlinks(tmp_path):
    root=tmp_path/'worktree';root.mkdir();(root/'local').mkdir()
    shared=tmp_path/'shared';shared.mkdir();artifact=shared/'pred.npy';artifact.write_bytes(b'synthetic fixture')
    (root/'local/runs').symlink_to(shared,target_is_directory=True)
    assert source_child(root,'local/runs/pred.npy').resolve()==artifact
    verify_hashes(root,{'local/runs/pred.npy':file_hash(artifact)},shared_runs=True)
    with pytest.raises(ValueError):source_child(root,'local/runs/../../outside')
    (root/'src').symlink_to(shared,target_is_directory=True)
    with pytest.raises(ValueError):source_child(root,'src/pred.npy')
    (shared/'escape').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):source_child(root,'local/runs/escape/outside')
    (root/'configs').mkdir();(root/'configs/data.local.yaml').symlink_to(artifact)
    assert source_child(root,'configs/data.local.yaml').resolve()==artifact
    (root/'configs/other.yaml').symlink_to(artifact)
    with pytest.raises(ValueError):source_child(root,'configs/other.yaml')
