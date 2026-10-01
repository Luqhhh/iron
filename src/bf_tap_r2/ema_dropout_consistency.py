"""Paired dropout regression regularization, retaining native EMA inference."""
import math
import resource

import numpy as np
import torch

from .component_regularization import ComponentRegressor, clone_state, inference_state, update_ema
from .v12_joint import joint_loss
from .v7_periodic import digest


def paired_dropout_loss(first, second, target, coefficient):
    if first.shape != second.shape or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Aligned predictions and finite nonnegative coefficient required')
    supervised = .5*(joint_loss(first, target)+joint_loss(second, target))
    consistency = (first-second).square().mean()
    loss = supervised+coefficient*consistency
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite paired dropout loss')
    return loss, consistency


def confirmation_eligible(gains, matched):
    if set(gains) != {'42', '3407'} or set(matched) != set(gains):
        raise ValueError('Two complete development splits required')
    values = [*gains.values(), *matched.values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Finite paired gains required')
    return min(values) > 0


class PairedDropoutEMARegressor(ComponentRegressor):
    def __init__(self, recipe, settings, arm, mechanisms, directory=None):
        coefficient = mechanisms.get('dropout_consistency_lambda')
        if arm != 'EMA' or coefficient not in (0., .5):
            raise ValueError('Only frozen paired-control or consistency EMA scope is supported')
        super().__init__(recipe, settings, arm, mechanisms, directory)

    def fit(self, frame, y):
        super().fit(frame, y)
        self.metadata_.update(training_mode='two_independent_dropout_forwards',
            dropout_consistency_lambda=self.mechanisms['dropout_consistency_lambda'])
        return self

    def _train(self, frame, y, epochs, validation=None):
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings['random_seed'])
        ema = clone_state(self.model_)
        coefficient = self.mechanisms['dropout_consistency_lambda']
        best, best_epoch, stale, best_state = float('inf'), 0, 0, None
        history = []
        if validation is not None:
            if set(frame.sample_id) & set(validation[0].sample_id):
                raise ValueError('Inner train/validation overlap')
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1]-self.mean_)/self.std_, dtype=torch.float32)
        for epoch in range(1, epochs+1):
            self.model_.train(); order = rng.permutation(len(frame))
            losses, discrepancies = [], []
            for start in range(0, len(order), self.settings['batch_size']):
                idx = order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                # Both arms advance the main dropout RNG twice; no RNG replay.
                first = self.model_(x[idx], cat[idx]); second = self.model_(x[idx], cat[idx])
                loss, discrepancy = paired_dropout_loss(first, second, target[idx], coefficient)
                loss.backward(); self.optimizer_.step()
                update_ema(self.model_, ema, self.mechanisms['ema_beta'])
                losses.append(float(loss.detach())); discrepancies.append(float(discrepancy.detach()))
            self.model_.eval()
            with inference_state(self.model_, ema), torch.no_grad():
                training_pred = self.model_(x, cat).mean(1)
                train_mae = float((training_pred-target).abs().mean())
                train_mse = float((training_pred-target).square().mean())
                value = float((self.model_(vx, vc).mean(1)-vy).abs().mean()) if validation is not None else None
            row = dict(epoch=epoch, updates=len(losses), gradient_evaluations=len(losses),
                training_forward_passes=2*len(losses), training_eval_mae=train_mae,
                training_eval_mse=train_mse, mean_batch_training_loss=float(np.mean(losses)),
                mean_prediction_consistency=float(np.mean(discrepancies)))
            if validation is not None:
                row['validation_mae'] = value
                if value < best-self.settings['min_delta']:
                    best, best_epoch, stale = value, epoch, 0
                    best_state = {key: tensor.clone() for key, tensor in ema.items()}
                else:
                    stale += 1
            history.append(row)
            if validation is not None and stale >= self.settings['patience']:
                break
        phase = 'selection' if validation is not None else 'refit'
        selected = best_epoch if validation is not None else epochs
        if validation is not None:
            if best_state is None: raise ValueError('No selected EMA state')
            self.selection_stopped_epoch_ = epoch
            self.model_.load_state_dict(best_state)
        else:
            self.model_.load_state_dict(ema)
        self.model_.eval()
        self.traces[phase] = dict(history=history, selected_epoch=selected, stopped_epoch=epoch,
            fit_ids_digest=digest(frame.sample_id.tolist()), fit_rows=len(frame),
            updates=sum(row['updates'] for row in history),
            gradient_evaluations=sum(row['gradient_evaluations'] for row in history),
            training_forward_passes=sum(row['training_forward_passes'] for row in history),
            dropout_consistency_lambda=coefficient,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if self.directory is not None:
            self.save(self.directory/f'{phase}.pt', self.traces[phase])
        return selected
