"""OPT-27/28 evidence freezing and zero-fit guards, scoped to v0.13."""
from contextlib import contextmanager, ExitStack
import sys
from pathlib import Path
from unittest.mock import patch
from ..artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment, verify_file_identities
from ..config import load_yaml
from ..exceptions import ContractError
from .component_export import forbid_fit, read_json
from .structural_run import append_ledger


@contextmanager
def zero_fit():
    from catboost import CatBoost
    from . import structural
    from .rate_model import RateModel
    from .recency_model import RecencyModel
    from .trajectory_candidate import TimeModel
    count = {'attempted_target_fits': 0, 'attempted_calibration_fits': 0,
             'completed_model_fits': 0, 'completed_calibration_fits': 0}
    def reject_model(*a, **k):
        count['attempted_target_fits'] += 1
        raise ContractError('v0.13 forbids model fitting')
    def reject_calibration(*a, **k):
        count['attempted_calibration_fits'] += 1
        raise ContractError('v0.13 forbids LAD/calibration fitting')
    with ExitStack() as stack:
        stack.enter_context(forbid_fit(count))
        stack.enter_context(patch.object(CatBoost, 'fit', reject_model))
        for model in (RateModel, RecencyModel, TimeModel):
            stack.enter_context(patch.object(model, 'fit', reject_model))
        original_functions = (structural.lad_coefficient, structural.fit_correction)
        # Guard already imported aliases in project modules, as well as original APIs.
        for module_name, module in list(sys.modules.items()):
            if module is not None and module_name.startswith('bf_tap.'):
                for name, value in list(vars(module).items()):
                    if any(value is fn for fn in original_functions):
                        stack.enter_context(patch.object(module, name, reject_calibration))
        yield count


def freeze(root, purpose, evidence, inputs=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    scope = load_yaml('configs/optimization_v0_13/access_scope.yaml')
    config = load_yaml('configs/optimization_v0_13/audit.yaml')
    protection = load_yaml(scope['protection_contract'])
    if not scope['holdout_consumed'] or not read_json('EVIDENCE_STATUS.json')['holdout_consumed']:
        raise ContractError('consumed retrospective authorization required')
    sources = [*Path('src/bf_tap').rglob('*.py'),
               *Path('configs/optimization_v0_13').glob('*.yaml'),
               *Path('scripts').glob('optimization_v13*.py'),
               *Path('tests').glob('test_v13*.py'),
               Path('docs/optimization_v0_13/PLAN.md'), Path('uv.lock')]
    manifest = {'purpose': purpose, 'config': config, 'scope': scope,
                'protection': protection, 'environment': runtime_environment(),
                'sources': file_identities({str(p): p for p in sources}),
                'inputs': file_identities(inputs or {}),
                'evidence': file_identities({str(p): p for p in evidence}),
                'old_ledgers': file_identities({p: p for p in scope['old_ledgers']}),
                'new_model_fits': 0, 'new_calibration_fits': 0,
                'test_inputs_for_selection': False, 'new_challengers': 0}
    atomic_write_json(root/'manifest.json', manifest)
    append_ledger({**scope, 'authorization': scope['authorization']+':'+purpose}, root)
    return manifest


def verify(manifest):
    for name in ('sources', 'inputs', 'evidence', 'old_ledgers'):
        verify_file_identities(manifest[name])


def verify_receipt(path):
    receipt = read_json(path)
    for name, digest in receipt['evidence_sha256'].items():
        if not Path(name).is_file():
            raise ContractError(f'BLOCKED_MISSING_SOURCE: {name}; expected SHA256 {digest}')
        if file_sha256(name) != digest:
            raise ContractError(f'prior completion evidence changed: {name}')
    return receipt
