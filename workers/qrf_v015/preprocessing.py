"""Train-only, schema-preserving input preparation for QRF_FULLTRAIN_LEAF_v1."""
import numpy as np


def validate(numeric, spout, ids, columns):
    numeric = np.asarray(numeric)
    if numeric.ndim != 2 or numeric.shape != (len(ids), len(columns)):
        raise ValueError('numeric/schema/ID alignment differs')
    if numeric.dtype.kind not in 'fiu' or np.isinf(numeric).any():
        raise ValueError('illegal numeric dtype or Inf')
    if len(spout) != len(ids) or len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError('duplicate, empty, or misaligned sample IDs')
    if len(set(columns)) != len(columns) or 'spout_no' in columns:
        raise ValueError('illegal numeric feature schema')
    return numeric.astype(np.float64)


class Preprocessor:
    def fit(self, numeric, spout, ids, columns):
        x = validate(numeric, spout, ids, columns)
        self.columns = list(columns)
        self.all_missing = np.isnan(x).all(axis=0)
        self.medians = np.array([0. if missing else np.nanmedian(x[:, j])
                                for j, missing in enumerate(self.all_missing)])
        self.vocabulary = sorted(set(str(s) for s in spout if str(s)))
        self.training_ids = list(ids)
        self.transformed_columns = [*self.columns,
            *[f'spout_no:known:{s}' for s in self.vocabulary],
            'spout_no:missing', 'spout_no:unknown']
        return self

    def transform(self, numeric, spout, ids, columns):
        x = validate(numeric, spout, ids, columns)
        if list(columns) != self.columns:
            raise ValueError('raw schema changed')
        mask = np.isnan(x)
        x = np.where(mask, self.medians, x)
        categories = {s: i for i, s in enumerate(self.vocabulary)}
        onehot = np.zeros((len(ids), len(categories)+2))
        indices = [categories.get(str(s), len(categories)+(1 if str(s) else 0)) for s in spout]
        onehot[np.arange(len(ids)), indices] = 1.
        joined = np.concatenate((x, onehot), axis=1)
        if not np.isfinite(joined).all() or (np.abs(joined) > np.finfo(np.float32).max).any():
            raise ValueError('non-finite or float32 overflow after imputation')
        return joined.astype(np.float32), {
            'imputed_fraction': float(mask.mean()) if mask.size else 0.,
            'missing_spout': sum(not str(s) for s in spout),
            'unknown_spout': sum(bool(str(s)) and str(s) not in categories for s in spout)}

    def metadata(self):
        return {'numeric_columns': self.columns, 'transformed_columns': self.transformed_columns,
                'medians': self.medians.tolist(), 'all_missing': self.all_missing.tolist(),
                'vocabulary': self.vocabulary, 'training_ids': self.training_ids}

    @classmethod
    def restore(cls, value):
        result = cls()
        result.columns = value['numeric_columns']
        result.transformed_columns = value['transformed_columns']
        result.medians = np.array(value['medians'])
        result.all_missing = np.array(value['all_missing'])
        result.vocabulary = value['vocabulary']
        result.training_ids = value['training_ids']
        return result
