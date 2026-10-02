"""Fixed Gaussian/Laplace locations on the unchanged V33 single-component net."""
import math
from pathlib import Path

import torch

from .laplace_time_model import LaplaceRegressor, laplace_nll
from .v33_mixture import MixtureRegressor

LOSS_ARMS = ('GAUSS_FIXED', 'LAPLACE_FIXED')


def fixed_gaussian_nll(y, location, scale):
    if y.ndim != 1 or location.shape != (len(y), 1):
        raise ValueError('Aligned scalar Gaussian location required')
    if not math.isfinite(scale) or scale <= 0 or not torch.isfinite(y).all() or not torch.isfinite(location).all():
        raise ValueError('Finite inputs and positive fixed scale required')
    return (.5 * ((y[:, None] - location) / scale).square()
            + math.log(scale) + .5 * math.log(2 * math.pi)).mean()


class FixedPointRegressor(LaplaceRegressor):
    # Reuse the exact, previously audited selector/refit training loop.
    def __init__(self, arm, settings):
        if arm not in LOSS_ARMS:
            raise ValueError('Unknown fixed point loss')
        MixtureRegressor.__init__(self, 'GAUSS1', settings)
        self.arm = arm

    def loss(self, y, output):
        _, location, scale_head = output
        if self.arm == 'GAUSS_FIXED':
            return fixed_gaussian_nll(y, location, self.settings['initial_scale'])
        return laplace_nll(y, location, torch.full_like(scale_head, self.settings['initial_scale']))

    def metadata(self):
        return dict(MixtureRegressor.metadata(self), point_loss_arm=self.arm,
                    fixed_scale=self.settings['initial_scale'], point_prediction='location')

    def save(self, path):
        payload = dict(recipe=self.recipe, point_loss_arm=self.arm,
            likelihood='fixed_point', settings=self.settings, preprocessing=self.preprocessor_.metadata(),
            mean=self.mean_, std=self.std_, state=self.model_.state_dict())
        with Path(path).open('xb') as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('likelihood') != 'fixed_point' or payload.get('point_loss_arm') not in LOSS_ARMS:
            raise ValueError('Explicit fixed point checkpoint identity required')
        base = MixtureRegressor.load(path)
        model = cls(payload['point_loss_arm'], base.settings)
        model.__dict__.update(base.__dict__)
        return model
