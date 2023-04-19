"""This module contains prediction models.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from eodfundeq import DatasetHelper
from eodfundeq.constants import ModelTypes

import pyfintools.tools.fts


class PredictionModel(ABC):
    @abstractmethod
    def predict(self, X):
        pass


class HeuristicSingleFeatureBull(PredictionModel):
    def __init__(self, ds_helper, feature_name):
        self.ds_helper = ds_helper
        self.feature_name = feature_name
        
    def predict(self, X):
        idx = self.ds_helper.feature_names.index(self.feature_name)
        return X[:,idx]        
    

class HeuristicSingleFeatureBear(PredictionModel):
    def __init__(self, ds_helper, feature_name):
        self.ds_helper = ds_helper
        self.feature_name = feature_name
        
    def predict(self, X):
        idx = self.ds_helper.feature_names.index(self.feature_name)
        return -X[:,idx]


class RandomModel(PredictionModel):
    def __init__(self, seed=1234):
        self._seed = seed
        self.rand_state = np.random.RandomState(self.seed)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, s):
        if self._seed != s:
            self._seed = s
            self.rand_state = np.random.RandomState(self.seed)

    def predict(self, X):
        N = X.shape[0]
        return self.rand_state.choice(N, N, replace=False)


class LGBMRankerAbstract(PredictionModel):
    @abstractmethod
    def get_features(self):
        pass

    @property
    @abstractmethod
    def direction(self):
        pass


class LGBMRankerMomentumAbstract(ABC):
    def __init__(self, feature_store, return_window=3, norm_returns=True,
                 exclude_nan_features=True, lgbm_kwargs=None):
        self.feature_store = feature_store
        self.return_window = return_window
        self.norm_returns = norm_returns
        self.exclude_nan_features = exclude_nan_features
        self.lgbm_kwargs = dict if lgbm_kwargs is None else lgbm_kwargs

        self.model = lgb.LGBMRanker(**lgbm_kwargs)
        self.momentum_windows = (3, 6, 12)
        self.excluded_features = ('u0', 'u1', 'u2')
        self._ds_helper = None

    # Implement abstract method
    def get_features(self):
        cta_mom_signals = self.feature_store.get_cta_momentum_signals()
        mom_signals = self._get_momentum_features()
        features = mom_signals | cta_mom_signals
        return {k: v for k, v in features.items() if k not in self.excluded_features}

    # Implement abstract method
    def predict(self, X):
        return self.model.predict(X)

    @property
    def ds_helper(self):
        if self._ds_helper is None:
            features = self.get_features()
            return DatasetHelper(self.feature_store, features,
                return_window=self.return_window, norm_returns=self.norm_returns,
                exclude_nan_features=self.exclude_nan_features)

    def fit(self):
        if self.ds_helper.norm_returns:
            y_vals = self.ds_helper.y_class_norm['train']
        else:
            y_vals = self.ds_helper.y_class['train']

        if self.direction == ModelTypes.BULL.value:
            args = (self.ds_helper.X['train'], y_vals)
        elif  self.direction == ModelTypes.BEAR.value:
            args = (self.ds_helper.X['train'], self.ds_helper.n_buckets - 1 - y_vals)
        else:
            raise NotImplementedError(f'Not supported for {self.direction}')

        kwargs = dict(
            group=self.ds_helper.groups['train'].flatten(), 
            feature_name=self.ds_helper.feature_names,
            eval_at=[100],
            eval_group=[[x] for x in self.ds_helper.groups['validation'].flatten()],
            eval_set=self.ds_helper.get_eval_set()[self.direction]
        )
        self.model.fit(*args, **kwargs)

    def _get_momentum_features(self):
        mom_features = dict()
        for window in self.momentum_windows:
            mom_array = self.feature_store.get_momentum(window)
            mom_features[f'mom_{window}m'] = mom_array

            vol = self.feature_store.get_volatility(window * 21, min_obs=window * 19)
            mom_array_norm = mom_array / np.clip(vol, 0.03, np.inf)
            mom_features[f'mom_{window}m_norm'] = mom_array_norm

            if window > 1:
                mom_array_lag = self.feature_store.get_momentum(window, lag=1)
                mom_features[f'mom_{window}m1_norm'] = mom_array_lag / np.clip(vol, 0.03, np.inf)
        return mom_features


class LGBMRankerMomentumBull(LGBMRankerMomentumAbstract):
    @property
    def direction(self):
        return ModelTypes.BULL.value


class LGBMRankerMomentumBear(LGBMRankerMomentumAbstract):
    @property
    def direction(self):
        return ModelTypes.BEAR.value