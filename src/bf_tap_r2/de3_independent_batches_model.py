"""Joint BASE TabM with independent, complete row permutations per head."""
import resource
import numpy as np
import torch

from .component_regularization import ComponentRegressor, clone_state
from .ema_independent_batches_model import member_orders, member_loss
from .v7_periodic import digest


class JointIndependentBatchRegressor(ComponentRegressor):
    def __init__(self, recipe, settings, directory=None):
        if settings.get('batch_order') != 'independent_without_replacement':
            raise ValueError('Frozen independent joint BASE training required')
        super().__init__(recipe, settings, 'BASE', {}, directory)

    def fit(self, frame, y):
        if np.asarray(y).shape != (len(frame), 2):
            raise ValueError('Exactly two native ordered joint targets required')
        return super().fit(frame, y)

    def _train(self, frame, y, epochs, validation=None):
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings['random_seed'])
        orders, history = [], []
        best, best_epoch, stale = float('inf'), 0, 0
        best_state = None
        if validation is not None:
            if set(frame.sample_id) & set(validation[0].sample_id):
                raise ValueError('Inner train/calibration overlap')
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1]-self.mean_)/self.std_, dtype=torch.float32)
        for epoch in range(1, epochs+1):
            self.model_.train()
            order = member_orders(rng, len(frame), self.settings['tabm_k'])
            orders.append(digest(order.tolist()))
            losses = []
            for start in range(0, len(order), self.settings['batch_size']):
                idx = order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = member_loss(self.model_(x[idx], cat[idx]), target[idx])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite joint independent-batch loss')
                loss.backward(); self.optimizer_.step()
                losses.append(float(loss.detach()))
            self.model_.eval()
            with torch.no_grad():
                prediction = self.model_(x, cat).mean(1)
                train_mae = float((prediction-target).abs().mean())
                train_mse = float((prediction-target).square().mean())
                value = float((self.model_(vx, vc).mean(1)-vy).abs().mean()) if validation is not None else None
            row = dict(epoch=epoch, updates=len(losses), gradient_evaluations=len(losses),
                training_eval_mae=train_mae, training_eval_mse=train_mse,
                mean_batch_training_loss=float(np.mean(losses)))
            if validation is not None:
                row['validation_mae'] = value
                if value < best-self.settings['min_delta']:
                    best, best_epoch, stale = value, epoch, 0
                    best_state = clone_state(self.model_)
                else:
                    stale += 1
            history.append(row)
            if validation is not None and stale >= self.settings['patience']:
                break
        phase = 'selection' if validation is not None else 'refit'
        selected = best_epoch if validation is not None else epochs
        if validation is not None:
            if best_state is None:
                raise ValueError('No finite selected joint state')
            self.selection_stopped_epoch_ = epoch
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.traces[phase] = dict(history=history, selected_epoch=selected, stopped_epoch=epoch,
            fit_ids_digest=digest(frame.sample_id.tolist()), fit_rows=len(frame),
            updates=sum(r['updates'] for r in history),
            gradient_evaluations=sum(r['gradient_evaluations'] for r in history),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            member_order_digests=orders, samples_per_head_per_epoch=len(frame),
            batch_order='independent_without_replacement')
        if self.directory is not None:
            self.save(self.directory/f'{phase}.pt', self.traces[phase])
        return selected
