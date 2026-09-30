from pathlib import Path

import pytest

from bf_tap_r2 import danet_phase_freeze as freeze
from bf_tap_r2.danet_ledger import file_hash,write_new
from test_danet_preflight import checked_workspace


def test_stale_suite_refuses_formal_freeze_before_resource_or_official_reference(checked_workspace,monkeypatch):
    workspace,path,_=checked_workspace;output=workspace/'local/not-created'
    (workspace/'src/bf_tap_r2/danet_network.py').write_text('# changed admitted architecture\n')
    def no_read(*args,**kwargs):raise AssertionError('Stale source reached resource or official reference')
    monkeypatch.setattr(freeze,'verify_transfer',no_read);monkeypatch.setattr(freeze,'load_reference',no_read)
    with pytest.raises(ValueError,match='exact-source'):
        freeze.prepare_manifest(workspace,output,path,file_hash(path),workspace,'0'*64,workspace,'0'*64)
    assert not output.exists()


def transferred_fixture(tmp_path,monkeypatch,status='passed'):
    old,new=tmp_path/'preserved',tmp_path/'formal'
    for root in (old,new):
        (root/'src').mkdir(parents=True);(root/'src/core.py').write_text('admitted original core\n')
    envelope=old/'local/danet-full-resource-r1';envelope.mkdir(parents=True)
    manifest=dict(workspace=str(old),runtime=freeze.runtime(),source_hashes={'src/core.py':file_hash(old/'src/core.py')})
    probe=envelope/'preflight';probe.mkdir();write_new(probe/'admission.json',{})
    report=dict(status=status,manifest_sha256=freeze.RESOURCE_MANIFEST_SHA,official_label_reads=0,official_estimator_fits=0,
        automatic_formal_execution=False,release_authorized=False,artifact_hashes={},required_available_mib=0)
    monkeypatch.setattr(freeze,'anchored',lambda path,sha:manifest if Path(path).name=='manifest.json' else report)
    return old,new,probe,report


def test_one_changed_transferred_source_cannot_consume_official_fit(tmp_path,monkeypatch):
    old,new,_,report=transferred_fixture(tmp_path,monkeypatch)
    assert freeze.verify_transfer(new,old,'0'*64)==report
    (new/'src/core.py').write_text('changed core\n')
    with pytest.raises(ValueError,match='transfer changed'):freeze.verify_transfer(new,old,'0'*64)


@pytest.mark.parametrize('defect',['failed_status','failure_file','unexpected_file','changed_artifact'])
def test_failed_or_changed_original_probe_blocks_the_formal_controller(tmp_path,monkeypatch,defect):
    old,new,probe,report=transferred_fixture(tmp_path,monkeypatch,status='failed' if defect=='failed_status' else 'passed')
    if defect=='failure_file':write_new(probe/'failed.json',{})
    elif defect=='unexpected_file':write_new(probe/'extra.json',{})
    elif defect=='changed_artifact':
        path=probe/'measurement.json';write_new(path,{})
        report['artifact_hashes'][path.name]=file_hash(path);path.write_text('changed evidence\n')
    with pytest.raises(ValueError):freeze.verify_transfer(new,old,'0'*64)
