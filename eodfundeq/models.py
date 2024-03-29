"""This module contains prediction models.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from abc import ABC, abstractmethod
from typing import Optional, Union

from eodfundeq import DatasetHelper
from eodfundeq.constants import DatasetTypes, ModelTypes

import pyfintools.tools.fts


class RollingForecastModel(object):
    def __init__(self, model, ds_helper, feature_names=None, clone_model=True, subsample=1.0):
        self.model = model
        self.ds_helper = ds_helper
        self.feature_names = feature_names
        self.clone_model = clone_model
        self.subsample = subsample

        self.model_list = []
        self.pred_list = []

    def fit(self):
        if self.model_list:
            return

        datagroup_iter = self.ds_helper.get_datagroup_iterator(sort=True)
        for _, datagroup in enumerate(datagroup_iter):
            if self.clone_model:
                period_model = sklearn.base.clone(self.model)
            else:
                period_model = self.model

            y_train = datagroup[DatasetTypes.TRAIN].y
            X_train = datagroup[DatasetTypes.TRAIN].X

            # Take a subsample
            n_samples = int(self.subsample * y_train.shape[0])
            idx = np.random.choice(y_train.shape[0], n_samples, replace=False)
            y_train = y_train.loc[idx]
            X_train = X_train.loc[idx]

            dg_test = datagroup[DatasetTypes.TEST]
            y_test = dg_test.y
            X_test = dg_test.X
            print(dg_test.timestamp.min(), dg_test.timestamp.max(), X_test.shape)
            timestamps = dg_test.timestamp.values
            symbols = dg_test.symbol.values
            if not y_test.size:
                continue

            if self.feature_names is not None:
                X_train = X_train[self.feature_names]
                X_test = X_test[self.feature_names]

            period_model.fit(X_train, y_train)
            y_test_pred = period_model.predict(X_test)

            df_period = pd.DataFrame(
                np.vstack([y_test.values, y_test_pred, timestamps, symbols]).T,
                index=y_test.index,
                columns=['true', 'pred', 'timestamp', 'symbol'])

            self.pred_list.append(df_period)
            self.model_list.append(period_model)

    def get_predictions(self):
        if not len(self.pred_list):
            raise ValueError('Must call "fit" method first to get predictions.')
        return pd.concat(self.pred_list, axis=0)


class PredictionModel(ABC):
    def __init__(self, feature_names: Union[str, list, None] = None):
        if isinstance(feature_names, str):
            self.feature_names = [feature_names]
        else:
            self.feature_names = feature_names

    @abstractmethod
    def predict(self, X):
        pass

    @property
    @abstractmethod
    def direction(self):
        pass


class HeuristicSingleFeatureAbstract(PredictionModel):
    def predict(self, X):
        assert len(self.feature_names) == 1, 'This class only allows a single feature'
        X_vals = X.loc[:,self.feature_names].values
        return X_vals if self.direction == ModelTypes.BULL else -X_vals


class HeuristicSingleFeatureBull(HeuristicSingleFeatureAbstract):
    @property
    def direction(self):
        return ModelTypes.BULL


class HeuristicSingleFeatureBear(HeuristicSingleFeatureAbstract):
    @property
    def direction(self):
        return ModelTypes.BEAR


class RandomModel(PredictionModel):
    def __init__(self, feature_names: Union[str, list, None] = None, seed=None):
        super().__init__(feature_names)
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

    @property
    def direction(self):
        return ModelTypes.BULL


class LGBMRankerAbstract(PredictionModel):
    def __init__(self, feature_names: Union[str, list, None] = None, lgbm_kwargs=None):
        super().__init__(feature_names=feature_names)
        self.lgbm_kwargs = dict if lgbm_kwargs is None else lgbm_kwargs
        self.model = lgb.LGBMRanker(**lgbm_kwargs)
        self.n_buckets = None

    # Implement abstract method
    def predict(self, X):
        X_prc = self._preprocess_features(X)
        return self.model.predict(X_prc)

    def fit(self, X, y, **kwargs):
        self.n_buckets = len(set(y))
        X_prc = self._preprocess_features(X)
        y_prc = self._preprocess_labels(y)
        kwargs_prc = kwargs.copy()
        if 'eval_set' in kwargs:
            kwargs_prc['eval_set'] = self._preprocess_eval_set(kwargs['eval_set'])
        self.model.fit(X_prc, y_prc, **kwargs_prc)

    def _preprocess_features(self, X):
        if self.feature_names is not None:
            X = X[self.feature_names].values
        return X

    def _preprocess_labels(self, y):
        y_prc = y
        if self.direction == ModelTypes.BEAR:
            y_prc = self.n_buckets - 1 - y
        elif self.direction != ModelTypes.BULL:
            raise ValueError(f'Unsupported model type: {self.direction}')
        return y_prc.values

    def _preprocess_eval_set(self, eval_set):
        eval_set_prc = []
        for tp in eval_set:
            X_orig, y_orig = tp
            X_adj = self._preprocess_features(X_orig)
            y_adj = y_orig
            if self.direction == ModelTypes.BEAR:
                y_adj = self.n_buckets - 1 - y_orig
            eval_set_prc.append((X_adj, y_adj))
        return eval_set_prc


class LGBMRankerBull(LGBMRankerAbstract):
    @property
    def direction(self):
        return ModelTypes.BULL


class LGBMRankerBear(LGBMRankerAbstract):
    @property
    def direction(self):
        return ModelTypes.BEAR


class BBModelAbstract(PredictionModel):
    def __init__(self, model, antag_model, fraction: float = 0.1, k: int = 100,
                 feature_names: Union[str, list, None] = None):
        super().__init__(feature_names=feature_names)
        self.model = model
        self.antag_model = antag_model
        self.fraction = fraction
        self.k = k

    def fit(self, X, y, **kwargs):
        pass

    # Implement abstract method
    def predict(self, X):
        mask = np.ones((X.shape[0],), dtype=bool)

        finished = False
        while not finished:
            y_score = self.model.predict(X)
            idx = y_score <= np.quantile(y_score[mask], self.fraction)
            if mask.sum() - mask[idx].sum() >= self.k:
                mask[idx] = False
            else:
                finished = True

            y_score[mask] = self.antag_model.predict(X[mask])
            idx = y_score >= np.quantile(y_score[mask], 1 - self.fraction)
            if mask.sum() - mask[idx].sum() >= self.k:
                mask[idx] = False
            else:
                finished = True

        y_score = self.model.predict(X)
        y_score[~mask] = -10e10
        assert y_score[mask].min() > -1e6
        assert mask.sum() >= self.k
        return y_score


class BBModelBull(BBModelAbstract):
    @property
    def direction(self):
        return ModelTypes.BULL


class BBModelBear(BBModelAbstract):
    @property
    def direction(self):
        return ModelTypes.BEAR


class MetaModelAbstract(LGBMRankerAbstract):
    def __init__(self, input_models, feature_names: list, lgbm_kwargs=None):
        super().__init__(feature_names=feature_names, lgbm_kwargs=lgbm_kwargs)
        self.input_models = input_models

    def _preprocess_features(self, X):
        input_preds = []
        for m in self.input_models:
            input_preds.append(m.predict(X))
        return pd.DataFrame(np.vstack(input_preds).T, 
                            columns=self.feature_names)

    def fit(self, X, y, **kwargs):
        kwargs_prc = kwargs.copy()
        if 'feature_name' in kwargs_prc:
            kwargs_prc['feature_name'] = self.feature_names
        return super().fit(X, y, **kwargs_prc)


class MetaModelBull(MetaModelAbstract):
    @property
    def direction(self):
        return ModelTypes.BULL


class MetaModelBear(MetaModelAbstract):
    @property
    def direction(self):
        return ModelTypes.BEAR


class SingleFeatureModel(object):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        
    # Implement this so that we can clone the model
    def get_params(self, **kwargs):
        return dict(feature_name=self.feature_name)

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return X[self.feature_name]
    
