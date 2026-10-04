"""Keep native MAE training callables serializable while observing validation."""
from contextlib import contextmanager
from unittest.mock import patch

from .realmlp_native_audit import observe_native


@contextmanager
def observe_mae_native(directory):
    from pytabkit.models.training.metrics import Metrics
    from pytabkit.models.training.nn_creator import NNCreator
    native_apply=Metrics.apply
    original_get=NNCreator.get_criterions

    def original_training_criterion(creator):
        # The author factory retains partial(Metrics.apply, metric_name='mae')
        # on the trained module. Never retain our local validation observer.
        with patch.object(Metrics,'apply',staticmethod(native_apply)):
            return original_get(creator)

    with observe_native(directory) as trace:
        with patch.object(NNCreator,'get_criterions',original_training_criterion):
            yield trace
