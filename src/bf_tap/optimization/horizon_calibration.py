"""OPT-30 single-time LAD and genuine calendar-horizon pairing; frozen bases."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json, file_sha256, stable_digest
from ..exceptions import ContractError
from .component_export import META, PRED, read_json, forbid_fit
from . import structural

CANDIDATE = 'V7_H2_MATCHED_TIME'
CONTROL = 'D1_H1_SAME_CALENDAR'
DATES = ['reference_time', 'label_available_at', 'fold_cutoff', 'train_reference_max',
         'train_available_max', 'history_available_max', 'evaluation_month_start', 'evaluation_month_end']


def month_start(month):
    if not isinstance(month, int) or not 1 <= month <= 12:
        raise ContractError('2024 calendar month required')
    return pd.Timestamp(year=2024, month=month, day=1, tz='Asia/Shanghai')


def calendar_window(fold_cutoff, horizon):
    c = pd.Timestamp(fold_cutoff)
    if c.tzinfo is None or horizon not in (1, 2):
        raise ContractError('aware cutoff and H1/H2 required')
    c = c.tz_convert('Asia/Shanghai')
    if c != month_start(c.month) or c.year != 2024:
        raise ContractError('original 2024 month-start cutoff required')
    return c + pd.DateOffset(months=horizon-1), c + pd.DateOffset(months=horizon)


def read_bank(path):
    from .dual_ratio_common import frame
    result = frame(path)
    for name in DATES:
        if name in result:
            result[name] = pd.to_datetime(result[name], utc=True).dt.tz_convert('Asia/Shanghai')
    return result


def verify_bank(bank, horizon, identities):
    required = {*META, *structural.INPUT, *DATES, 'horizon', 'fold_identity_sha256'}
    if required-set(bank) or bank.empty or bank.sample_id.duplicated().any() or bank[list(required)].isna().any().any():
        raise ContractError('complete unique forecast bank required')
    if {'tap_iron', 'tap_time_len'} & set(bank):
        raise ContractError('forecast bank must not contain targets')
    if not (bank.horizon == horizon).all():
        raise ContractError('forecast horizon differs')
    for cutoff, g in bank.groupby('fold_cutoff', sort=True):
        start, end = calendar_window(cutoff, horizon)
        key = str(pd.Timestamp(cutoff).month)
        if key not in identities:
            raise ContractError('unregistered fold')
        identity = identities[key]
        if (pd.Timestamp(identity['fit_cutoff']) != cutoff or
                not (g.fold_identity_sha256 == stable_digest(identity)).all()):
            raise ContractError('forecast cutoff/model identity mismatch')
        expected = {'train_reference_max': identity['reference_max'],
                    'train_available_max': identity['label_available_max'],
                    'history_available_max': identity['history_available_max']}
        if any(not (g[k] == pd.Timestamp(v)).all() for k, v in expected.items()):
            raise ContractError('forecast temporal certificate mismatch')
        if not ((g.reference_time >= start) & (g.reference_time < end) &
                (g.evaluation_month_start == start) & (g.evaluation_month_end == end) &
                (g.train_reference_max < cutoff) & (g.train_available_max <= cutoff) &
                (g.history_available_max <= cutoff) & (g.label_available_at >= g.reference_time)).all():
            raise ContractError('forecast calendar or availability leakage')
    structural.directions(bank[structural.INPUT])
    return True


def verify_pair(h2, h1):
    if h2.sample_id.duplicated().any() or h1.sample_id.duplicated().any():
        raise ContractError('duplicate matched sample')
    a, b = [x.set_index('sample_id').sort_index() for x in (h2, h1)]
    cols = ['spout_no', 'reference_time', 'label_available_at', 'evaluation_month_start', 'evaluation_month_end']
    if not a.index.equals(b.index) or not a[cols].equals(b[cols]):
        raise ContractError('matched H1/H2 calendar identity differs')
    if not ((a.horizon == 2).all() and (b.horizon == 1).all()):
        raise ContractError('matched bank horizons differ')
    return True


def select_matched(h2, h1, history, cutoff, minimum=100):
    verify_pair(h2, h1)
    c = pd.Timestamp(cutoff)
    if c.tzinfo is None or history.sample_id.duplicated().any():
        raise ContractError('aware outer cutoff and unique history required')
    if (history.reference_time >= c).any() or (history.available_at > c).any():
        raise ContractError('outer history exceeds cutoff')
    keep = ((h2.evaluation_month_end <= c) & (h2.reference_time < c) &
            (h2.fold_cutoff < c) & (h2.label_available_at <= c))
    selected = h2.loc[keep].sort_values(['reference_time', 'sample_id'], kind='mergesort')
    if selected.empty or len(selected) < minimum:
        raise ContractError('BLOCKED_INSUFFICIENT_MATCHED_OOF')
    labels = history.set_index('sample_id')
    if not set(selected.sample_id) <= set(labels.index):
        raise ContractError('calibration label absent from certified outer history')
    label_rows = labels.loc[selected.sample_id]
    if (not np.array_equal(label_rows.spout_no.astype(str), selected.spout_no.astype(str)) or
            not np.array_equal(label_rows.reference_time, selected.reference_time) or
            not np.array_equal(label_rows.available_at, selected.label_available_at)):
        raise ContractError('matched label metadata differs from certified history')
    out = []
    for bank in (h2, h1):
        x = bank.set_index('sample_id').loc[selected.sample_id].reset_index()
        for target in ('tap_iron', 'tap_time_len'):
            x[target] = label_rows[target].to_numpy()
        x = structural.select_oof(x, c, minimum)
        out.append(x)
    if not np.array_equal(out[0].sample_id, out[1].sample_id) or not np.array_equal(out[0][['tap_iron','tap_time_len']], out[1][['tap_iron','tap_time_len']]):
        raise ContractError('matched fitting identities differ')
    return tuple(out)


def predict_time(inputs, original_v1, beta):
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ContractError('frozen scalar coefficient in [0,1] required')
    if list(original_v1) != ['sample_id', *PRED] or original_v1.sample_id.duplicated().any():
        raise ContractError('original complete V1 predictions required')
    delta, usable = structural.directions(inputs)
    v = original_v1.set_index('sample_id')
    if set(v.index) != set(inputs.sample_id):
        raise ContractError('V1/input identity differs')
    values = v.loc[inputs.sample_id, PRED].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError('finite original V1 required')
    result = inputs[['sample_id']].copy()
    result[PRED[0]] = values[:, 0].copy()
    result[PRED[1]] = np.maximum(0., inputs[PRED[1]].to_numpy()+beta*delta[:, 1])
    return result, int((~usable).sum())


def lad_certificate(selected, beta):
    """Verify the stored scalar's optimum and smallest-tie rule without fitting."""
    direction, _ = structural.directions(selected[structural.INPUT]); d = direction[:, 1]
    y, base = selected.tap_time_len.to_numpy(float), selected[PRED[1]].to_numpy(float)
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ContractError('invalid stored coefficient')
    residual = base+beta*d-y
    kink = np.abs(residual) <= 1e-10
    gradient = float(np.sum(d[~kink]*np.sign(residual[~kink])))
    radius = float(np.sum(np.abs(d[kink])))
    lo, hi = gradient-radius, gradient+radius
    optimum = hi >= -1e-8 if beta == 0 else lo <= 1e-8 if beta == 1 else lo <= 1e-8 and hi >= -1e-8
    nz = d != 0
    if not nz.any():
        smallest = beta == 0
    else:
        ratios = (y[nz]-base[nz])/d[nz]; weights = np.abs(d[nz])
        order = np.argsort(ratios, kind='mergesort'); ratios, weights = ratios[order], weights[order]
        cumulative = np.cumsum(weights); half = .5*weights.sum()
        # Inspect masses around the given coefficient; do not compute a new coefficient.
        below = cumulative[ratios < beta]
        at_or_below = cumulative[ratios <= beta]
        mass_below = float(below[-1]) if len(below) else 0.
        mass_at_or_below = float(at_or_below[-1]) if len(at_or_below) else 0.
        smallest = (mass_at_or_below >= half if beta == 0 else
                    mass_below < half if beta == 1 else
                    mass_below < half and mass_at_or_below >= half)
        if beta == 0: smallest = optimum  # The constrained left endpoint is minimal.
    if not optimum or not smallest:
        raise ContractError('stored LAD optimum/smallest-tie certificate failed')
    return {'beta': float(beta), 'rows': len(selected), 'ids_sha256': stable_digest(selected.sample_id.tolist()),
            'labels_sha256': stable_digest(selected[['sample_id','tap_time_len']].to_dict('records')),
            'objective': float(np.abs(residual).sum()), 'subgradient_left': lo,
            'subgradient_right': hi, 'smallest_minimizer_verified': True}


class ScalarBudget:
    """Persistent one-attempt-per-role/origin; successful coefficients are reusable."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def fit(self, role, month, selected):
        if role not in (CANDIDATE, CONTROL) or month not in range(6,12):
            raise ContractError('unregistered LAD target/origin')
        key = f'{role}-{month}'
        intent, result = self.root/f'{key}.intent.json', self.root/f'{key}.json'
        identity = stable_digest(selected.to_dict('records'))
        if intent.exists():
            if not result.exists() or read_json(intent)['selected_sha256'] != identity:
                raise ContractError('LAD already attempted; restore evidence without refitting')
            saved = read_json(result)
            if saved['certificate'] != lad_certificate(selected, saved['beta']):
                raise ContractError('saved coefficient evidence differs')
            return saved
        attempts = list(self.root.glob(f'{role}-*.intent.json'))
        if len(attempts) >= 6:
            raise ContractError('LAD budget exceeded')
        # Exclusive creation accounts for the attempt before entering the scalar primitive.
        import json
        with intent.open('x') as f:
            json.dump({'role':role,'month':month,'selected_sha256':identity,'target':'tap_time_len'}, f)
        d, _ = structural.directions(selected[structural.INPUT])
        beta = structural.lad_coefficient(selected.tap_time_len, selected[PRED[1]], d[:,1])
        saved = {'role':role,'month':month,'beta':beta,'selected_sha256':identity,
                 'certificate':lad_certificate(selected,beta)}
        atomic_write_json(result, saved)
        return saved

    def counts(self):
        return {role:{'attempted':len(list(self.root.glob(f'{role}-*.intent.json'))),
                      'completed':len(list(self.root.glob(f'{role}-*.json')))-len(list(self.root.glob(f'{role}-*.intent.json')))}
                for role in (CANDIDATE,CONTROL)}


@contextmanager
def no_tree_or_dual_fit():
    from catboost import CatBoost
    from .rate_model import RateModel
    count = {'attempted_target_fits':0,'attempted_dual_calibration_fits':0}
    def reject_model(*args, **kwargs):
        count['attempted_target_fits'] += 1
        raise ContractError('OPT30 forbids tree/rate model fits')
    def reject_dual(*args, **kwargs):
        count['attempted_dual_calibration_fits'] += 1
        raise ContractError('OPT30 forbids implicit two-target correction fits')
    import sys
    original = structural.fit_correction
    with ExitStack() as stack:
        stack.enter_context(forbid_fit(count))
        stack.enter_context(patch.object(CatBoost,'fit',reject_model))
        stack.enter_context(patch.object(RateModel,'fit',reject_model))
        for module_name,module in list(sys.modules.items()):
            if module is not None and module_name.startswith('bf_tap.'):
                for name,value in list(vars(module).items()):
                    if value is original: stack.enter_context(patch.object(module,name,reject_dual))
        yield count


def acceptance(metrics, summary, reg, engineering, iron_exact):
    a = reg['acceptance']; cells = {}
    for h in range(1,5):
        values = [v['candidates'] for k,v in metrics.items() if k.startswith('O') and v['horizon'] == h]
        cells[h] = np.array([v[CANDIDATE]['overall']['loss']-v['V1']['overall']['loss'] for v in values])
    h2units = {k:v for k,v in metrics.items() if k.startswith('O') and v['horizon']==2}
    recent = [v['candidates'][CANDIDATE]['overall']['loss']-v['candidates']['V1']['overall']['loss']
              for k,v in h2units.items() if int(k[5:7])+1 in a['recent_H2_evaluation_months']]
    control_delta = float(np.mean([v['candidates'][CANDIDATE]['overall']['loss']-v['candidates'][CONTROL]['overall']['loss'] for v in h2units.values()]))
    j = summary[CANDIDATE]['J']-summary['V1']['J']
    dev = {k:metrics[k]['candidates'][CANDIDATE]['overall']['loss']-metrics[k]['candidates']['V1']['overall']['loss'] for k in ('DEV_LONG','DEV_SHORT')}
    gates = {'engineering_causal_identity':bool(engineering),'iron_exact':bool(iron_exact),
             'H2_mean_E':float(cells[2].mean()) <= a['H2_E_delta_max'],
             'H2_improved_origins':int((cells[2]<0).sum()) >= a['H2_min_strict_improved'],
             'recent_H2_improved':sum(x<0 for x in recent) >= a['recent_H2_min_strict_improved'],
             'H2_single_origin':float(cells[2].max()) <= a['H2_single_E_delta_max'],
             'H2_same_calendar_control':control_delta <= a['H2_E_minus_D1_max'],
             'J_guardrail':j <= a['J_delta_max']}
    gates.update({f'H{h}_guardrail':float(cells[h].mean()) <= a['H134_E_delta_max'] for h in (1,3,4)})
    gates.update({f'{k}_guardrail':v <= a['DEV_E_delta_max'] for k,v in dev.items()})
    passed = all(gates.values())
    return {'status':reg['pass_status'] if passed else reg['failure_action'],'gates':gates,
            'H2_delta_E':float(cells[2].mean()),'H2_improved':int((cells[2]<0).sum()),
            'recent_H2_improved':sum(x<0 for x in recent),'H2_single_max_delta_E':float(cells[2].max()),
            'H2_V7_minus_D1':control_delta,'delta_J':j,
            'horizon_delta_E':{str(h):float(v.mean()) for h,v in cells.items()},'DEV_delta_E':dev,
            'engineering_valid':bool(engineering),'historical_quality_passed':passed,
            'official_data_identity_verified':False,'source_semantics_verified_or_original_assumption_retained':'ORIGINAL_ASSUMED_CONTRACT_RETAINED',
            'final_coefficient_fitted':False,'ready_challenger':False,'platform_verified':False,
            'diagnostic_control_releasable':False}


def require_official_identity(state):
    if not state.get('historical_quality_passed') or not state.get('official_data_identity_verified'):
        raise ContractError('final coefficient/challenger blocked until quality and official identity pass')
