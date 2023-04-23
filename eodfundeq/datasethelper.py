import numpy as np
import pandas as pd

from typing import Optional

from eodfundeq.constants import DatasetTypes, ModelTypes
from eodfundeq import utils


class TrainingIntervals(object):
    def __init__(self, data_start='1999-12-31', data_end='2022-12-31', train_start='2003-12-31',
        num_embargo_periods=12, n_months_valid=36, n_months_test=18):

        # Parameters needed to split train/validation/test sets
        self.data_start = pd.Timestamp(data_start)
        self.data_end = pd.Timestamp(data_end)
        self.train_start = pd.Timestamp(train_start)
        self.num_embargo_periods = num_embargo_periods
        self.n_months_valid = n_months_valid
        self.n_months_test = n_months_test
        self._temp_filename = None

    def get_intervals(self):
        intervals = dict()
        test_start = self.data_end - pd.tseries.offsets.MonthEnd(self.n_months_test)
        intervals[DatasetTypes.TEST] = (test_start, self.data_end)
        valid_end = test_start - pd.tseries.offsets.MonthEnd(self.num_embargo_periods)
        valid_start = valid_end - pd.tseries.offsets.MonthEnd(self.n_months_valid)
        intervals[DatasetTypes.VALIDATION] = (valid_start, valid_end)
        train_end = valid_start - pd.tseries.offsets.MonthEnd(self.num_embargo_periods)
        intervals[DatasetTypes.TRAIN] = (self.train_start, train_end)
        return intervals

    
class Dataset(object):
    def __init__(self,
                 X: pd.DataFrame, 
                 y_reg: Optional[pd.DataFrame] = None,
                 y_cls: Optional[pd.DataFrame] = None,
                 y_pct: Optional[pd.DataFrame] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 true_return: Optional[pd.DataFrame] = None):
        self.X = X
        self.y_reg = y_reg if y_reg is not None else pd.DataFrame()
        self.y_cls = y_cls if y_cls is not None else pd.DataFrame()
        self.y_pct = y_pct if y_pct is not None else pd.DataFrame()        
        self.metadata = metadata if metadata is not None else pd.DataFrame()
        self.true_return = true_return if true_return is not None else pd.DataFrame()

    @property
    def timestamp(self):
        return self.metadata.timestamp

    @property
    def symbol(self):
        return self.metadata.symbol


class DataGroup(object):
    def __init__(self, DStrain: Dataset, DSvalidation: Dataset, DStest: Dataset):
        self.data = {DatasetTypes.TRAIN: DStrain,
                     DatasetTypes.VALIDATION: DSvalidation,
                     DatasetTypes.TEST: DStest}

    def X(self):
        return {dst: self.data[dst].X for dst in DatasetTypes}

    def y_reg(self):
        return {dst: self.data[dst].y_reg for dst in DatasetTypes}

    def y_cls(self):
        return {dst: self.data[dst].y_cls for dst in DatasetTypes}

    def y_pct(self):
        return {dst: self.data[dst].y_pct for dst in DatasetTypes}

    def metadata(self):
        return {dst: self.data[dst].metadata for dst in DatasetTypes}

    def symbol(self):
        return {dst: self.data[dst].symbol for dst in DatasetTypes}

    def timestamp(self):
        return {dst: self.data[dst].timestamp for dst in DatasetTypes}

    def true_return(self):
        return {dst: self.data[dst].true_return for dst in DatasetTypes}


class DatasetHelper(object):
    def __init__(self, feature_store, training_intervals, features_dict=None,
                 return_window=3, norm_returns=True, exclude_nan_features=False, n_buckets=5):
        self.feature_store = feature_store
        self._training_intervals = training_intervals
        self.return_window = return_window
        self.norm_returns = norm_returns
        self.exclude_nan_features = exclude_nan_features
        self.n_buckets = n_buckets
        self._features_dict = dict() if features_dict is None else features_dict
        self._volatility = dict()

    @property
    def features_dict(self):
        return self._features_dict

    @features_dict.setter
    def features_dict(self, fd):
        assert isinstance(fd, dict), 'Can only set features_dict with variables of type dict.'
        self._features_dict = fd

    @property
    def feature_names(self):
        return sorted(self.features_dict.keys())

    @property
    def metadata_names(self):
        return ['timestamp', 'symbol']

    @property
    def training_intervals(self):
        return self._training_intervals

    @training_intervals.setter
    def training_intervals(self, ti):
        assert isinstance(ti, TrainingIntervals), 'Unsupported input type for training_intervals.'
        self._training_intervals = ti

    @property
    def labels(self):
        pass

    @property
    def features(self):
        pass

    def get_dataframe(self):
        df_base = self._get_base_dataframe()        
        df = self._add_dataset_type(df_base)
        return self._add_bucketed_returns(df)

    def get_datagroup(self):
        df = self.get_dataframe()
        ds_group_args = dict()
        for ds_type in DatasetTypes:
            sub_df = df.loc[df.dataset_type.values == ds_type.value]
            ds_group_args[ds_type] = Dataset(X=sub_df[self.feature_names],
                                             y_reg=sub_df.y_reg,
                                             y_cls=sub_df.y_cls,
                                             y_pct=sub_df.y_pct,
                                             metadata=sub_df[self.metadata_names],
                                             true_return=sub_df.true_return)
        return ds_group_args

    def _get_volatility(self, n_months):
        if n_months not in self._volatility:
            if n_months == 1:
                n_days, min_obs = 63, 60
            elif n_months == 3:
                n_days, min_obs = 126, 120
            elif n_months == 6:
                n_days, min_obs = 252, 240                
            else:
                raise NotImplementedError('The volatility window corresponding to this return window has not been chosen.')

            self._volatility[n_months] = self.feature_store.get_volatility(n_days, min_obs=min_obs)
        return self._volatility[n_months]

    def _get_returns(self):
        future_returns = self.feature_store.get_future_returns(self.return_window)
        mask = self.feature_store.good_mask
        true_rtn = pd.Series(future_returns[mask], name='true_return')
        if not self.norm_returns:
            y = future_returns.copy()
        else:
            volatility = self._get_volatility(self.return_window)
            y = future_returns / np.maximum(volatility, 0.03)
        y_reg = pd.Series(y[mask], name='y_reg')
        assert y_reg.size == true_rtn.size, 'Returns are of different length'
        return pd.concat([true_rtn, y_reg], axis=1)         

    def _get_features(self):
        mask = self.feature_store.good_mask
        features_df_list = []
        for feature_name in self.feature_names:
            df = pd.Series(self.features_dict[feature_name][mask],
                           name=feature_name)
            features_df_list.append(df)
        assert len(set([x.size for x in features_df_list])) == 1, 'Features are of different lengths'
        return pd.concat(features_df_list, axis=1)

    def _get_metadata(self):
        mask = self.feature_store.good_mask
        dates = self.feature_store.dates['m'].values
        date_panel = np.tile(dates.reshape(-1, 1), len(self.feature_store.symbols))
        df_timestamp = pd.Series(date_panel[mask], name='timestamp')
        symbol_panel = np.tile(self.feature_store.symbols.reshape(-1, 1), len(dates)).T
        df_symbols = pd.Series(symbol_panel[mask], name='symbol')
        metadata_list = [df_timestamp, df_symbols]
        assert len(set([x.size for x in metadata_list])) == 1, 'Data is of different lengths'
        return pd.concat(metadata_list, axis=1)

    def _get_base_dataframe(self):
        df_features = self._get_features()
        df_returns = self._get_returns()
        df_metadata = self._get_metadata()
        assert df_features.shape[0] == df_returns.shape[0]
        assert df_features.shape[0] == df_metadata.shape[0]
        df = pd.concat([df_features, df_returns, df_metadata], axis=1)
        idx_no_nan = ~np.isnan(df_returns.true_return) & ~np.isnan(df_returns.y_reg)
        if self.exclude_nan_features:
            idx_no_nan = idx_no_nan & np.all(~np.isnan(df_features), axis=1)
        df = df.loc[idx_no_nan, :]
        df.sort_values(['timestamp', 'symbol'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    def _add_dataset_type(self, df):
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df['dataset_type'] = ''
        intervals = self.training_intervals.get_intervals()
        for dataset_type, interval in intervals.items():
            start, end = interval
            df.loc[start:end, 'dataset_type'] = dataset_type.value        
        return df.reset_index()

    def _add_bucketed_returns(self, df):
        df['y_cls'] = -1      # Add new column for classification problems
        df['y_pct'] = np.nan  # Add new column containing the percentile of the return
        for tmstmp in set(df.timestamp.values):
            idx_t = df.timestamp.values == tmstmp
            if not np.any(idx_t):
                continue

            df.loc[idx_t, 'y_pct'] = utils.calc_feature_percentiles(df.y_reg.loc[idx_t], axis=0)
            edges = np.linspace(0, 1, self.n_buckets+1)[:-1]
            df.loc[idx_t, 'y_cls'] = -1 + np.digitize(df.y_pct.loc[idx_t], edges)
        return df

    def get_lgbmranker_groups(self, dataset):
        # Get the # of samples in each group (aka the # of stocks for each date)
        assert dataset.timestamp.is_monotonic_increasing, 'Samples should be sorted by timestamp'
        df_t = dataset.timestamp.reset_index()
        return df_t.groupby('timestamp').count().sort_index().to_numpy().flatten()

    def get_lgbmranker_eval_set(self, dataset, feature_names):
        eval_set = []
        timestamps = dataset.timestamp.values
        for t in sorted(set(timestamps)):
            idx_t = (t == timestamps)
            X = dataset.X.loc[t == timestamps, feature_names]
            y = dataset.y_cls.loc[t == timestamps]
            eval_set.append((X, y))
        return eval_set