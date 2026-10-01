"""Freeze and execute the zero-fit review, keeping vectors in private local/."""
import argparse
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK/'src'))


def main():
    from bf_tap_r2.q75_combination_review import evaluate
    from bf_tap_r2.ema_evaluation_diagnostics import sha, write_new
    from bf_tap_r2.v5_library import load_v5_training_frame, fold_vector
    from bf_tap_r2.v5_spec import load_v5_spec
    from bf_tap_r2.v49_run import check_runtime
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-root', default='/home/lux1/iron')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main_root, out = Path(args.main_root).resolve(), Path(args.output).resolve()
    if out.exists() or not out.is_relative_to(main_root/'local/runs'):
        raise ValueError('Fresh private run directory required')
    spec_path = WORK/'configs/q75_combination_review/SPEC.json'
    spec = json.loads(spec_path.read_text())
    check_runtime(spec)
    current = json.loads((main_root/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if any(current.get(k) != v for k, v in spec['reference'].items()):
        raise ValueError('Reference changed before review')
    files = {str(p): sha(p) for p in [spec_path, Path(__file__),
        WORK/'src/bf_tap_r2/q75_combination_review.py', WORK/'docs/q75_combination_review/PREREGISTRATION.md']}
    # Historical frozen receipts already bind source and model identities.
    for name in (spec['old_development'], spec['confirmation_cache']):
        directory = main_root/name
        for path in directory.rglob('*'):
            if path.is_file() and path.name in ('manifest.json', 'audit.json', 'complete.json',
                    'warm-complete.json', 'cold-complete.json', 'metadata.json', 'predictions.npz'):
                files[str(path)] = sha(path)
    data = json.loads((main_root/spec['old_development']/'manifest.json').read_text())['data_hashes']
    files.update({str(main_root/p): h for p, h in data.items()})
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Frozen cache/data changed: '+p)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'manifest.json', dict(spec=spec, files=files,
        authorization='user_requested_sparse_EMA_inclusive_review', new_fits=0))
    frame = load_v5_training_frame(main_root)
    folds = {s: fold_vector(main_root, frame, s, load_v5_spec(main_root)) for s in spec['split_seeds']}
    report = evaluate(main_root, frame, folds, spec, out)
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Frozen cache/data changed during review: '+p)
    print(json.dumps(dict(status=report['status'], selected=report['selected_on_development'],
        four_seed_gate=report['selected_four_seed_gate'], summaries=report['seed_summaries'])), flush=True)


if __name__ == '__main__':
    main()
