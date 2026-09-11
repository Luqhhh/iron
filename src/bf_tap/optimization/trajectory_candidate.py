"""Single direct-duration model and fixed V1-iron-preserving OPT-24 correction."""
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from ..artifacts import atomic_write_json, file_sha256
from ..exceptions import ContractError
from ..models.baseline import FROZEN_PARAMETERS
from ..features.trajectory import COLUMNS
from .component_export import PRED, read_json
from .rate_model import schema
from .structural import INPUT, directions, lad_coefficient, select_oof

CANDIDATE = 'V5_TIME_TRAJECTORY'


class TimeModel:
    def fit(self, x, duration, budget):
        y = np.asarray(duration, dtype=float)
        if list(x)[-24:] != COLUMNS or y.shape != (len(x),) or not len(y) or not np.isfinite(y).all() or (y < 0).any():
            raise ContractError('invalid registered direct-time training inputs')
        self.schema = schema(x)
        self.model = CatBoostRegressor(**FROZEN_PARAMETERS)
        budget.allow(self.model)
        self.model.fit(x, y, cat_features=['spout_no'])
        return self

    def predict(self, x):
        if schema(x) != self.schema:
            raise ContractError('trajectory model schema changed')
        values = np.asarray(self.model.predict(x), dtype=float)
        if not np.isfinite(values).all():
            raise ContractError('nonfinite direct time prediction')
        return np.maximum(values, 0.)

    def save(self, root, metadata):
        root = Path(root); root.mkdir(parents=True, exist_ok=False)
        self.model.save_model(root/'time.cbm')
        atomic_write_json(root/'bundle.json', dict(candidate=CANDIDATE, target='tap_time_len',
            parameters=FROZEN_PARAMETERS, feature_schema=self.schema, training=metadata,
            model_sha256=file_sha256(root/'time.cbm')))

    @classmethod
    def load(cls, root):
        root = Path(root); md = read_json(root/'bundle.json')
        if md['candidate'] != CANDIDATE or md['target'] != 'tap_time_len' or md['parameters'] != FROZEN_PARAMETERS or file_sha256(root/'time.cbm') != md['model_sha256']:
            raise ContractError('trajectory bundle differs')
        obj = cls(); obj.schema = md['feature_schema']; obj.metadata = md
        obj.model = CatBoostRegressor(); obj.model.load_model(root/'time.cbm')
        return obj


def fit_time_coefficient(oof, cutoff, minimum=100, rate_floor=1e-6):
    selected = select_oof(oof, cutoff, minimum)
    delta, _ = directions(selected[INPUT], rate_floor)
    return lad_coefficient(selected.tap_time_len, selected.pred_tap_time_len, delta[:, 1]), selected


def predict(parts, original_v1, alpha, rate_floor=1e-6):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ContractError('time LAD coefficient outside [0,1]')
    delta, usable = directions(parts[INPUT], rate_floor)
    if list(original_v1) != ['sample_id', *PRED] or original_v1.sample_id.duplicated().any() or set(original_v1.sample_id) != set(parts.sample_id):
        raise ContractError('V1 identity mismatch')
    out = original_v1.set_index('sample_id').loc[parts.sample_id].reset_index().copy()
    out[PRED[1]] = np.maximum(0., parts[PRED[1]].to_numpy() + alpha*delta[:, 1])
    if not np.array_equal(out[PRED[0]], original_v1.set_index('sample_id').loc[out.sample_id, PRED[0]]) or not np.isfinite(out[PRED].to_numpy()).all():
        raise ContractError('frozen iron/nonfinite prediction violation')
    return out, int((~usable).sum())


def acceptance(metrics, summary, reg, iron_exact, engineering):
    policy = reg['acceptance']; current, reference = summary[CANDIDATE], summary['V1']
    delta = {h: {t: current['horizons'][h][t]-reference['horizons'][h][t]
                 for t in ('mean_loss', 'iron_mean_wmape', 'time_mean_wmape')}
             for h in ('H1', 'H2', 'H3', 'H4')}
    origins = {u: v['candidates'][CANDIDATE]['overall']['loss']-v['candidates']['V1']['overall']['loss']
               for u, v in metrics.items() if v['horizon'] == 1}
    if len(origins) != 6 or len([v for v in metrics.values() if v['horizon'] is not None]) != 18:
        raise ContractError('registered six-origin 18-cell grid required')
    dev = {u: metrics[u]['candidates'][CANDIDATE]['overall']['loss']-metrics[u]['candidates']['V1']['overall']['loss']
           for u in ('DEV_LONG', 'DEV_SHORT')}
    checks = dict(iron_exact=bool(iron_exact), engineering_and_causal=bool(engineering),
        J=current['J']-reference['J'] <= policy['J_delta_max'],
        H1_time=delta['H1']['time_mean_wmape'] <= policy['H1_time_delta_max'],
        H1_origins=sum(d < 0 for d in origins.values()) >= policy['H1_min_strictly_improved_origins'],
        recent_H1=sum(origins[f'O2024{m:02d}_H1'] < 0 for m in policy['recent_H1_months']) >= policy['recent_H1_min_improved'],
        **{h: delta[h]['mean_loss'] <= policy['H234_E_delta_max'] for h in ('H2', 'H3', 'H4')},
        **{u: d <= policy['DEV_E_delta_max'] for u, d in dev.items()})
    passed = all(checks.values())
    return dict(candidate=CANDIDATE, reference='V1', passed=passed, checks=checks,
        selected=CANDIDATE if passed else None, J=current['J'], J_delta_vs_V1=current['J']-reference['J'],
        horizon_deltas=delta, H1_origin_deltas=origins, DEV_deltas=dev,
        status='PASS' if passed else 'FAIL_CLOSE_REGISTERED_V5')
