"""Native TabM-packed with the frozen EMA selector/fresh-refit procedure."""
from pathlib import Path

import numpy as np
import torch
from rtdl_num_embeddings import PeriodicEmbeddings
from tabm import TabM

from .component_regularization import ComponentRegressor
from .data import FEATURES
from .v3_6_networks import NumericPreprocessor


def make_packed_network(recipe, settings, categories, outputs):
    if recipe != {'backbone': 'tabm', 'frequency': .01} or settings.get('arch_type') != 'tabm-packed':
        raise ValueError('Frozen native TabM-packed architecture required')
    embedding = PeriodicEmbeddings(len(FEATURES), d_embedding=settings['embedding_dim'],
        n_frequencies=settings['n_frequencies'], frequency_init_scale=recipe['frequency'],
        lite=settings['lite'])
    return TabM.make(n_num_features=len(FEATURES), cat_cardinalities=[categories],
        d_out=outputs, k=settings['tabm_k'], n_blocks=settings['blocks'],
        d_block=settings['width'], dropout=settings['dropout'], num_embeddings=embedding,
        arch_type='tabm-packed', start_scaling_init=None)


class PackedEMARegressor(ComponentRegressor):
    def __init__(self, recipe, settings, arm, mechanisms, directory=None):
        if arm != 'EMA' or settings.get('arch_type') != 'tabm-packed' or 'batch_order' in settings:
            raise ValueError('Native packed EMA with original shared batches required')
        super().__init__(recipe, settings, arm, mechanisms, directory)

    def _initialize(self, frame, y):
        if np.asarray(y).shape != (len(frame), 1):
            raise ValueError('Exactly one time target required')
        torch.set_num_threads(1)
        torch.manual_seed(self.settings['random_seed'])
        self.preprocessor_ = NumericPreprocessor(structure='raw_tabm').fit(frame)
        self.mean_, self.std_ = np.mean(y, axis=0), np.std(y, axis=0)
        if not (self.std_ > 0).all():
            raise ValueError('Constant target')
        self.model_ = make_packed_network(self.recipe, self.settings,
            self.preprocessor_.n_spout_categories_, y.shape[1])
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
            lr=self.settings['learning_rate'], weight_decay=self.settings['weight_decay'])

    @classmethod
    def load(cls, path):
        saved = torch.load(Path(path), map_location='cpu', weights_only=True)
        result = cls(saved['recipe'], saved['settings'], saved['arm'], saved['mechanisms'])
        p = NumericPreprocessor(structure='raw_tabm')
        info = saved['preprocessing']
        p.means_, p.stds_ = np.asarray(info['means']), np.asarray(info['stds'])
        p.spout_to_index_ = {int(k): v for k, v in info['spout_vocabulary'].items()}
        p.n_spout_categories_ = info['n_spout_categories']
        result.preprocessor_ = p
        result.mean_, result.std_ = np.asarray(saved['mean']), np.asarray(saved['std'])
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            result.model_ = make_packed_network(result.recipe, result.settings,
                p.n_spout_categories_, len(result.mean_))
        result.model_.load_state_dict(saved['state'], strict=True)
        result.model_.eval()
        result.saved = saved
        return result
