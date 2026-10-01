"""Package existing Q75 evidence and a prospective SWA scope without any fits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from verify_swa_continuation_handoff import child, require, sha, verify_bundle


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def package(repo, output, checks_receipt):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    require(output.is_relative_to(repo / 'local') and not output.exists(),
            'Use a fresh private output directory; no overwrite')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip(),
            'Publishable source must be committed before packaging')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'Set numeric threads before imports')
    require(sys.version_info[:2] == (3, 12), 'Use locked Python3.12')
    checks = json.loads(Path(checks_receipt).read_text())
    require(checks['status'] == 'passed' and checks['new_scientific_fits'] == checks['label_values_read'] == 0,
            'Successful zero-fit checks required')
    spec_path = repo / 'configs/tabm_time_swa_continuation/CONFIRMATION_SPEC.json'
    spec = json.loads(spec_path.read_text())
    binding_path = repo / 'local/research/q75-existing-reference-export-r1/binding.json'
    require(sha(binding_path) == spec['reference_export_binding_sha256'], 'Frozen export changed')
    binding = json.loads(binding_path.read_text())
    original = repo / 'configs/tabm_time_swa_v1/SPEC.json'
    require(sha(original) == spec['original_spec_sha256'], 'Original protocol changed')
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / 'payload'
    bundle.mkdir()
    mapping = {}

    def copy(source, name, expected=None, original_path=None):
        source = Path(source)
        if expected is not None:
            require(sha(source) == expected, 'Source changed before packaging: ' + str(source))
        dest = child(bundle, name)
        require(not dest.exists(), 'Duplicate payload path')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        require(sha(dest) == sha(source), 'Copy changed bytes')
        if original_path is not None:
            mapping[original_path] = name

    copy(spec_path, 'CONFIRMATION_SPEC.json')
    copy(original, 'ORIGINAL_SWA_SPEC.json')
    copy(repo / 'docs/tabm_time_swa_continuation/HANDOFF.md', 'README.md')
    copy(repo / 'docs/tabm_time_swa_continuation/PREREGISTRATION.md', 'PREREGISTRATION.md')
    copy(binding_path, 'refs/original-binding.json')
    exported = [binding['ids']]
    for column in binding['columns'].values():
        exported.extend(column.values())
    for item in exported:
        copy(item['path'], 'refs/' + Path(item['path']).name, item['sha256'], item['path'])
    main = Path('/home/lux1/iron')
    for name, digest in binding['frozen_original_evidence'].items():
        source = Path(name)
        require(source.is_relative_to(main / 'local/runs'), 'Unexpected private evidence scope')
        copy(source, 'evidence/' + str(source.relative_to(main)), digest, name)
    for name, digest in spec['source_hashes'].items():
        copy(child(repo, name), 'source_snapshot/' + name, digest)
    for source in sorted((repo / 'docs/tabm_time_swa_v1').glob('*')):
        if source.is_file():
            copy(source, 'original_records/' + source.name)
    for name in ('prepare_swa_confirmation_references.py', 'verify_swa_continuation_handoff.py',
                 'package_swa_continuation_handoff.py'):
        copy(repo / 'scripts' / name, 'scripts/' + name)
    for name in ('test_swa_confirmation_reference_export.py', 'test_swa_continuation_handoff.py',
                 'test_tabm_swa_window.py', 'test_tabm_swa_integration.py'):
        copy(repo / 'tests' / name, 'tests/' + name)
    copy(checks_receipt, 'receipts/handoff-checks.json')
    for name in ('pytest.log', 'initial-import-path-failure.json'):
        copy(Path(checks_receipt).parent / name, 'receipts/' + name)
    copy(repo / 'local/research/swa-continuation-checks-r1/receipt.json', 'receipts/reference-export-checks.json')
    files = {str(p.relative_to(bundle)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(bundle.rglob('*')) if p.is_file()}
    manifest = dict(identity='SWA_CONFIRMATION_PRIVATE_HANDOFF_20261001_R1',
                    purpose='private_research_handoff_not_platform_submission',
                    branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip(),
                    published_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    original_source_commit=spec['original_source_commit'],
                    confirmation_spec_sha256=sha(spec_path),
                    original_reference_binding_sha256=sha(binding_path),
                    payload_files=files, original_path_to_payload=mapping,
                    direct_original_evidence_files=len(binding['frozen_original_evidence']),
                    original_teammate_SWA_development_artifacts_included=False,
                    full_reference_mother_model_cold_dependencies_included=False,
                    official_data_csvs_included=False, new_scientific_fits=0,
                    parsed_label_values=0, scientific_execution_started=False,
                    release_authorized=False, platform_submission_packages=0, uploads=0)
    write_new(bundle / 'HANDOFF_MANIFEST.json', manifest)
    digest = sha(bundle / 'HANDOFF_MANIFEST.json')
    verify_bundle(bundle, digest)
    archive = output / 'SWA_confirmation_271828_314159_handoff_20261001.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                info = zipfile.ZipInfo(str(p.relative_to(bundle)))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, p.read_bytes())
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'Archive CRC failed')
        require(len(z.namelist()) == len(set(z.namelist())) == len(files) + 1, 'Archive inventory differs')
        for name in z.namelist():
            child(bundle, name)
        extracted = output / 'cold-extracted-r1'
        extracted.mkdir(exist_ok=False)
        z.extractall(extracted)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with (output / 'cold-portable-verification.log').open('x') as log:
        subprocess.run([sys.executable, str(extracted / 'scripts/verify_swa_continuation_handoff.py'),
                        '--directory', str(extracted), '--expected-manifest-sha256', digest],
                       cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / (archive.name + '.sha256')).open('x') as stream:
        stream.write(sha(archive) + '  ' + archive.name + '\n')
    with (output / 'HANDOFF_MANIFEST.sha256').open('x') as stream:
        stream.write(digest + '  HANDOFF_MANIFEST.json\n')
    with (output / '先读此文件.txt').open('x') as stream:
        stream.write('请将ZIP与两个sha256文件一起发给队友。此为SWA私有实验交接包。\n'
                     '解压到新目录，先看README.md。新确认为271828/314159；原42/3407开发和7777/12011协议保留。\n'
                     '接收方核验已有开发和原账本后，另行闭合运行清单再执行确认。生产方未启动拟合。\n'
                     '参照四seed均完整2754行/五折，直接来源证据98份。冷进程搬移核验与ZIP CRC通过。\n'
                     'manifest SHA256: ' + digest + '\n'
                     'ZIP SHA256: ' + sha(archive) + '\n')
    receipt = dict(status='passed_private_handoff_package_and_independent_portable_verification',
                   archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                   manifest_sha256=digest, payload_files=len(files) + 1,
                   direct_original_evidence_files=len(binding['frozen_original_evidence']),
                   confirmation_spec_sha256=sha(spec_path),
                   published_source_commit=manifest['published_source_commit'],
                   cold_verification_log_sha256=sha(output / 'cold-portable-verification.log'),
                   new_scientific_fits=0, label_values_read=0, scientific_execution_started=False,
                   platform_submission_packages=0, uploads=0, original_protocol_changed=False)
    write_new(output / 'packaging-receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', required=True)
    parser.add_argument('--checks-receipt', required=True)
    args = parser.parse_args()
    package(args.repo, args.output, args.checks_receipt)
