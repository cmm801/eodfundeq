import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Optional

from eodfundeq.constants import DatasetTypes, ForecastTypes, ModelTypes, TSNames, TRADING_DAYS_PER_YEAR
from eodfundeq import utils

from eodfundeq.filters import EqualFilter, InRangeFilter, EntireColumnInRangeFilter, IsNotNAFilter


class TrainingIntervalsAbstract(ABC):
    """Abstract class for getting train/validation/test intervals."""

    @abstractmethod
    def get_all_intervals(self):
        """Returns a dict with a map from Dataset type to start/end date tuple."""

    @property
    def is_rolling(self):
        return False

    def get_intervals(self, dataset_type):
        """Provide the start/end date for a specified dataset."""
        return self.get_all_intervals()[dataset_type]

    def get_train_intervals(self):
        """Provide the start/end date for the training dataset."""
        return self.get_intervals(DatasetTypes.TRAIN)

    def get_validation_intervals(self):
        """Provide the start/end date for the validation dataset."""
        return self.get_intervals(DatasetTypes.VALIDATION)

    def get_test_intervals(self):
        """Provide the start/end date for the test dataset."""
        return self.get_intervals(DatasetTypes.TEST)


class TrainingIntervalsByPeriod(TrainingIntervalsAbstract):
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

    # Implement abstract method
    def get_all_intervals(self):
        intervals = dict()
        test_start = self.data_end - pd.tseries.offsets.MonthEnd(self.n_months_test)
        intervals[DatasetTypes.TEST] = (test_start, self.data_end)
        valid_end = test_start - pd.tseries.offsets.MonthEnd(self.num_embargo_periods)
        valid_start = valid_end - pd.tseries.offsets.MonthEnd(self.n_months_valid)
        intervals[DatasetTypes.VALIDATION] = (valid_start, valid_end)
        train_end = valid_start - pd.tseries.offsets.MonthEnd(self.num_embargo_periods)
        intervals[DatasetTypes.TRAIN] = (self.train_start, train_end)
        return intervals


class TrainingIntervalsCustom(TrainingIntervalsAbstract):
    """Class that provides custom specified train/validation/test intervals"""

    def __init__(self, start_train, end_train, start_valid, end_valid,
                 start_test, end_test):
        self.start_train = pd.Timestamp(start_train)
        self.end_train = pd.Timestamp(end_train)
        self.start_valid = pd.Timestamp(start_valid)
        self.end_valid = pd.Timestamp(end_valid)
        self.start_test = pd.Timestamp(start_test)
        self.end_test = pd.Timestamp(end_test)

    # Implement abstract method
    def get_all_intervals(self):
        """Returns a dict with a map from Dataset type to start/end date tuple."""
        return {
            DatasetTypes.TRAIN: (self.start_train, self.end_train),
            DatasetTypes.VALIDATION: (self.start_valid, self.end_valid),
            DatasetTypes.TEST: (self.start_test, self.end_test),
        }


class RollingTrainingIntervals(TrainingIntervalsCustom):
    """Class that produces a set of rolling test/train intervals."""
    def __init__(self, start_train, end_train, end_test, fit_freq='y', expanding=True):
        super().__init__(start_train=start_train, end_train=end_train,
                         start_valid=end_train, end_valid=end_train,
                         start_test=end_train, end_test=end_test)
        self.fit_freq = fit_freq
        self.expanding = expanding

    @property
    def is_rolling(self):
        return True

    # Implement abstract method
    def get_all_intervals(self):
        """Returns a list of dicts mapping from Dataset type to start/end date tuple."""        
        date_offset = utils.get_date_offset(frequency=self.fit_freq)
        period_ends = pd.date_range(self.end_train, self.end_test - date_offset,
                                    freq=self.fit_freq)
        if not self.expanding:
            training_window = self.end_train - self.start_train

        rolling_intervals = []
        start_train = self.start_train
        for end_train in period_ends:
            if not self.expanding:
                start_train = end_train - training_window

            interval = TrainingIntervalsCustom(
                start_train=start_train, end_train=end_train,
                start_valid=end_train, end_valid=end_train,
                start_test=end_train, end_test=end_train + date_offset)
            rolling_intervals.append(interval.get_all_intervals())
        return rolling_intervals


class Dataset(object):
    def __init__(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None,
                 metadata: Optional[pd.DataFrame] = None, is_sorted=False):
        self.is_sorted = False
        self.X = X
        self.y = y if y is not None else pd.DataFrame()
        self.metadata = metadata if metadata is not None else pd.DataFrame()
        self.is_sorted = is_sorted

    @property
    def timestamp(self):
        return self.metadata.timestamp

    @property
    def symbol(self):
        return self.metadata.symbol

    def sort_by_timestamp(self):
        if not self.is_sorted:
            sort_idx = np.argsort(self.timestamp)
            self.X = self.X.loc[sort_idx]
            self.y = self.y.loc[sort_idx]
            self.metadata = self.metadata.loc[sort_idx]
            self.is_sorted = True

    def get_cv_group_index(self, n_splits=5):
        bin_edges = np.linspace(0, 1, n_splits+1)[1:-1]
        if not self.is_sorted:
            self.sort_by_timestamp()
        uniq_timestamps = sorted(set(self.timestamp.drop_duplicates().values))
        timestamps = self.timestamp.values
        bin_edges = np.linspace(0, len(uniq_timestamps), n_splits+1)[1:-1].astype(int)
        groups = n_splits * np.ones_like(timestamps, dtype=int)
        for j, bin_edge in enumerate(bin_edges[::-1]):
            groups[timestamps <= uniq_timestamps[bin_edge]] = n_splits - j - 1
        return groups


class DataGroup(object):
    def __init__(self, DStrain: Dataset, DSvalidation: Dataset, DStest: Dataset):
        self.data = {DatasetTypes.TRAIN: DStrain,
                     DatasetTypes.VALIDATION: DSvalidation,
                     DatasetTypes.TEST: DStest}

    def X(self):
        return {dst: self.data[dst].X for dst in DatasetTypes}

    def y(self):
        return {dst: self.data[dst].y for dst in DatasetTypes}

    def metadata(self):
        return {dst: self.data[dst].metadata for dst in DatasetTypes}

    def symbol(self):
        return {dst: self.data[dst].symbol for dst in DatasetTypes}

    def timestamp(self):
        return {dst: self.data[dst].timestamp for dst in DatasetTypes}


class DatasetHelper(object):
    def __init__(self, feature_store, training_intervals, forecast_horizon,
                 features_dict=None, frequency='m',
                 forecast_type=ForecastTypes.NORM_RETURNS.value,
                 exclude_nan_features=False, n_buckets=0, min_data_fraction=0.9):
        self.feature_store = feature_store
        self.frequency = frequency
        self._training_intervals = training_intervals
        self.forecast_horizon = forecast_horizon
        self.forecast_type = forecast_type
        self.exclude_nan_features = exclude_nan_features
        self.n_buckets = n_buckets
        self.min_data_fraction = min_data_fraction
        self._features_dict = {} if features_dict is None else features_dict
        self.volatility_tol = 0.01  # lower bound on vol to prevent taking Log(0)

        # Initialize dict to cache volatility calculations
        self._volatility_cache = {}

        # Initialize mask that will tell us if data points are valid
        self._good_mask = None
        self._filters = []

        self.filter_min_obs = 10        # Excludes dates/metrics with few observations
        self.filter_min_price = 1       # Excludes stocks with too low of a price
        self.filter_min_monthly_volume = 21 * 10000  # Exclude stocks with low trading volume
        self.filter_max_return = 10     # Excludes return outliers from sample

        # Set default filters
        self.set_default_filters()

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
        assert isinstance(ti, TrainingIntervalsAbstract), 'Unsupported input type for training_intervals.'
        self._training_intervals = ti

    @property
    def filters(self):
        return self._filters

    @property
    def good_mask(self):
        if self._good_mask is None:
            self._good_mask = self.feature_store.init_pd_dataframe(
                frequency=self.frequency, dtype=bool)
            for f in self.filters:
                self._good_mask &= f.get_mask()

        return self._good_mask.values

    @property
    def symbols(self):
        return self.feature_store.symbols

    def get_dataframe(self, sort=False):
        df = self._get_base_dataframe()
        df.set_index('timestamp', inplace=True)
        return df.sort_index() if sort else df

    def get_datagroup(self, sort=False):
        if self.training_intervals.is_rolling:
            raise ValueError('This method is only supported for non-rolling training intervals.')
        return next(self.get_datagroup_iterator(sort=sort))

    def get_datagroup_iterator(self, sort=False):
        df = self.get_dataframe(sort=sort)
        if self.training_intervals.is_rolling:
            rolling_intervals = self.training_intervals.get_all_intervals()
        else:
            rolling_intervals = [self.training_intervals.get_all_intervals()]

        date_offset = utils.get_date_offset(self.frequency)
        for rolling_interval in rolling_intervals:
            ds_group_args = {}
            for dataset_type, interval in rolling_interval.items():
                start, end = interval

                # Add offset to start so that first date is not included
                sub_df = df.loc[start+date_offset:end].reset_index()
                ds_group_args[dataset_type] = Dataset(
                    X=sub_df[self.feature_names], y=sub_df.y,
                    metadata=sub_df[self.metadata_names], is_sorted=sort)
            yield ds_group_args

    def set_default_filters(self):
        self._good_mask = None
        if self.frequency == 'm':
            self.set_default_filters_monthly()
        if self.frequency == 'b':
            self.set_default_filters_daily()
        
    def set_default_filters_monthly(self):
        fs = self.feature_store
        self._filters = [
            IsNotNAFilter(fs, TSNames.CLOSE.value),
            IsNotNAFilter(fs, TSNames.ADJUSTED_CLOSE.value),
            IsNotNAFilter(fs, TSNames.ADJUSTED_CLOSE.value,
                          property_func=lambda x: x.rolling(12).mean()),
            InRangeFilter(fs, TSNames.DAILY_PRICES.value, high=3, high_inc=True,
                          property_func=lambda x: np.isnan(x).rolling(63).sum().resample('M').last()),
            IsNotNAFilter(fs, TSNames.VOLUME.value),
            InRangeFilter(fs, TSNames.CLOSE.value,
                          low=self.filter_min_price),
            InRangeFilter(fs, TSNames.ADJUSTED_CLOSE.value,
                          low=self.filter_min_price),
            InRangeFilter(fs, TSNames.VOLUME.value, 
                          property_func=lambda x: x.rolling(12).quantile(0.01, interpolation='lower'),
                          low=self.filter_min_monthly_volume),
            EntireColumnInRangeFilter(fs, TSNames.ADJUSTED_CLOSE.value, high=self.filter_max_return,
                                      property_func=lambda x: -1 + x / x.shift(1).values),
        ]
        
    def set_default_filters_daily(self):
        fs = self.feature_store        
        self._filters = [
            IsNotNAFilter(fs, TSNames.DAILY_PRICES.value),
            InRangeFilter(fs, TSNames.DAILY_PRICES.value,
                          low=self.filter_min_price),
            InRangeFilter(fs, TSNames.DAILY_PRICES.value,
                          property_func=lambda x: x.rolling(63, min_periods=1).min(),
                          low=self.filter_min_price),
            InRangeFilter(fs, TSNames.DAILY_PRICES.value, high=self.filter_max_return,
                          property_func=lambda x: x.pct_change())
        ]

    def reset_default_filters(self):
        self.set_default_filters()

    def add_filter(self, f):
        self._good_mask = None
        self._filters.append(f)

    def _get_volatility_forecast(self, n_months):
        if n_months not in self._volatility_cache:
            if n_months == 1:
                n_days, min_obs = 63, 60
            elif n_months == 3:
                n_days, min_obs = 126, 120
            elif n_months == 6:
                n_days, min_obs = 252, 240
            else:
                raise NotImplementedError('The volatility window corresponding to this return window has not been chosen.')

            self._volatility_cache[n_months] = self.feature_store.get_volatility(n_days, min_obs=min_obs)
        return np.maximum(self._volatility_cache[n_months], 0.03)

    def flatten_panel(self, panel):
        """Takes a data panel and flattens it into a single-column."""
        return panel[self.good_mask]

    def _get_labels(self):
        if self.forecast_type == ForecastTypes.VOLATILITY.value:
            y_panel = self._get_volatility_labels()
        elif self.forecast_type == ForecastTypes.LOG_VOLATILITY.value:
            y_panel = np.log(np.maximum(self._get_volatility_labels(), self.volatility_tol))
        elif self.forecast_type in (ForecastTypes.RETURNS.value, ForecastTypes.NORM_RETURNS.value):
            y_panel = self._get_return_labels()
        else:
            raise ValueError(f'Unsupported forecast type: {self.forecast_type}')
        return pd.DataFrame(self.flatten_panel(y_panel), columns=['y'])

    def _get_volatility_labels(self):
        return self.feature_store.get_future_realized_volatility(
            forecast_horizon=self.forecast_horizon,
            min_fraction=self.min_data_fraction,
            frequency=self.frequency)

    def _get_return_labels(self):
        y_panel = self.feature_store.get_future_returns(self.forecast_horizon)
        if self.forecast_type == ForecastTypes.NORM_RETURNS.value:
            volatility = self._get_volatility_forecast(self.forecast_horizon)
            y_panel /= volatility

        if self.n_buckets > 0:
            y_panel = utils.bucket_features(y_panel, self.n_buckets, axis=1)
        return y_panel

    def add_feature_panel(self, feature_name, feature_panel):
        self.features_dict[feature_name] = self.flatten_panel(feature_panel)

    def _get_features(self):
        features_df_list = []
        for feature_name in self.feature_names:
            df = pd.Series(self.features_dict[feature_name], name=feature_name)
            features_df_list.append(df)
        if len(set([x.size for x in features_df_list])) != 1:
            raise ValueError('Features are of different lengths')
        df_features = pd.concat(features_df_list, axis=1)
        return df_features

    def create_panel_from_time_series(self, ts):
        """Create a data panel with copies of a time series."""
        return np.tile(ts.values.reshape(-1, 1), self.symbols.size)

    def _get_metadata(self):
        dates = self.feature_store.dates[self.frequency].values
        date_panel = self.feature_store.create_panel_from_time_series(dates)
        df_timestamp = pd.Series(self.flatten_panel(date_panel), name='timestamp')
        symbol_panel = self.feature_store.create_panel_from_row(
            self.feature_store.symbols, frequency=self.frequency)
        df_symbols = pd.Series(self.flatten_panel(symbol_panel), name='symbol')
        metadata_list = [df_timestamp, df_symbols]
        assert len(set([x.size for x in metadata_list])) == 1, 'Data is of different lengths'
        return pd.concat(metadata_list, axis=1)

    def _get_base_dataframe(self):
        df_features = self._get_features()
        df_labels = self._get_labels()
        df_metadata = self._get_metadata()
        assert df_features.shape[0] == df_labels.shape[0]
        assert df_features.shape[0] == df_metadata.shape[0]
        df = pd.concat([df_features, df_labels, df_metadata], axis=1)
        idx_no_nan = ~np.isnan(df_labels.y)
        if self.exclude_nan_features:
            idx_no_nan = idx_no_nan & np.all(~np.isnan(df_features), axis=1)
        df = df.loc[idx_no_nan, :]
        df.sort_values(['timestamp', 'symbol'], inplace=True)
        df.reset_index(inplace=True, drop=True)
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
            X = dataset.X.loc[t == timestamps, feature_names]
            y = dataset.y.loc[t == timestamps]
            eval_set.append((X, y))
        return eval_set
