import numpy as np
import pandas as pd

from eodfundeq.constants import START_DATE, DataSetTypes
from eodfundeq import filters
from eodfundeq import utils


BULL = 'bull'
BEAR = 'bear'


class DatasetHelper():
    def __init__(self, featureObj, features_dict, return_window=3):
        self.featureObj = featureObj
        self.features_dict = features_dict
        self.return_window = return_window
        self._filter_max_monthly_volume = np.inf
        self.reset_cache()

        self.target_vol = 0.30
        self.exclude_nan = False
        
        # Parameters needed to split train/validation/test sets
        self.data_start = pd.Timestamp('1999-12-31')
        self.data_end = pd.Timestamp('2022-12-31')
        self.train_start = pd.Timestamp('2003-12-31')
        self.num_insulation_periods = 12
        self.n_months_valid = 36
        self.n_months_test = 18        
        
    def get_intervals(self):
        intervals = dict()
        test_start = self.data_end - pd.tseries.offsets.MonthEnd(self.n_months_test)
        intervals[DataSetTypes.TEST.value] = (test_start, self.data_end)
        valid_end = test_start - pd.tseries.offsets.MonthEnd(self.num_insulation_periods)
        valid_start = valid_end - pd.tseries.offsets.MonthEnd(self.n_months_valid)
        intervals[DataSetTypes.VALIDATION.value] = (valid_start, valid_end)
        train_end = valid_start - pd.tseries.offsets.MonthEnd(self.num_insulation_periods)
        intervals[DataSetTypes.TRAIN.value] = (self.train_start, train_end)
        return intervals
        
    @property
    def features_dict(self):
        return self._features_dict
    
    @features_dict.setter
    def features_dict(self, fd):
        self._features_dict = fd
        self.reset_cache()
    
    @property
    def filter_max_monthly_volume(self):
        return self._filter_max_monthly_volume
    
    @filter_max_monthly_volume.setter
    def filter_max_monthly_volume(self, mv):
        if self._filter_max_monthly_volume != mv:
            self._filter_max_monthly_volume = mv
            self._datasets = None

    @property
    def feature_names(self):
        return sorted(self.features_dict.keys())

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = dict(X=dict(), year_month=dict(), symbol=dict(),
                                  y_class=dict(), y_reg=dict(), y_pct=dict(),
                                  y_class_norm=dict(), y_reg_norm=dict(), y_pct_norm=dict(),
                                 )
            dataset_masks = self.get_dataset_masks()
            dataset_features_lists = self._get_dataset_features_lists()
            future_returns = self.featureObj.get_future_returns(self.return_window)
            if self.return_window == 1:
                volatility = self.featureObj.get_volatility(63, min_obs=57)
            elif self.return_window == 3:
                volatility = self.featureObj.get_volatility(126, min_obs=115)
            else:
                raise NotImplementedError('The volatility window corresponding to this return window has not been chosen.')

            n_symbols = len(self.featureObj.symbols)
            dates = self.featureObj.dates['m']
            ym_int = np.array([x.year * 100 + x.month for x in dates])
            ym_panel = np.tile(ym_int.reshape(-1, 1), n_symbols)
            symbol_panel = np.tile(np.arange(n_symbols).reshape(-1, 1), len(dates)).T

            for dataset_type, mask in dataset_masks.items():
                fut_rtns = future_returns.copy()
                fut_rtns[~mask] = np.nan
                bucketed_returns = self.featureObj.get_buckets(fut_rtns)
                pred_pct = utils.calc_feature_percentiles(bucketed_returns)
                pred_bkt = np.digitize(pred_pct, np.linspace(0, 1, self.featureObj.n_buckets+1)[:-1])
                y_0 = pred_bkt[mask] - 1

                fut_rtns_norm = fut_rtns / np.clip(volatility, 0.03, np.inf)
                bucketed_returns_norm = self.featureObj.get_buckets(fut_rtns_norm)
                pred_pct_norm = utils.calc_feature_percentiles(bucketed_returns_norm)
                pred_bkt_norm = np.digitize(pred_pct_norm, np.linspace(0, 1, self.featureObj.n_buckets+1)[:-1])
                y_0_norm = pred_bkt_norm[mask] - 1                

                X_raw = np.vstack(dataset_features_lists[dataset_type]).T
                idx_no_nan = ~np.isnan(fut_rtns[mask]) & ~np.isnan(fut_rtns_norm[mask])
                if self.exclude_nan:
                    idx_no_nan = idx_no_nan & np.all(~np.isnan(X_raw), axis=1)
                self._datasets['X'][dataset_type] = X_raw[idx_no_nan,:]
                self._datasets['y_class'][dataset_type] = y_0[idx_no_nan].astype(int)
                self._datasets['y_reg'][dataset_type] = fut_rtns[mask][idx_no_nan]
                self._datasets['y_pct'][dataset_type] = pred_pct[mask][idx_no_nan]
                self._datasets['y_class_norm'][dataset_type] = y_0_norm[idx_no_nan].astype(int)
                self._datasets['y_reg_norm'][dataset_type] = fut_rtns_norm[mask][idx_no_nan]
                self._datasets['y_pct_norm'][dataset_type] = pred_pct_norm[mask][idx_no_nan]
                self._datasets['year_month'][dataset_type] = ym_panel[mask][idx_no_nan]
                self._datasets['symbol'][dataset_type] = symbol_panel[mask][idx_no_nan]
                assert ~np.any(np.isnan(fut_rtns[mask][idx_no_nan]))
        return self._datasets

    @property
    def X(self):
        return self.datasets['X']

    @property
    def y_class(self):
        return self.datasets['y_class']

    @property
    def y_reg(self):
        return self.datasets['y_reg']

    @property
    def y_pct(self):
        return self.datasets['y_pct']

    @property
    def y_class_norm(self):
        return self.datasets['y_class_norm']

    @property
    def y_reg_norm(self):
        return self.datasets['y_reg_norm']

    @property
    def y_pct_norm(self):
        return self.datasets['y_pct_norm']    

    @property
    def symbol(self):
        return self.datasets['symbol']

    @property
    def year_month(self):
        return self.datasets['year_month']

    def reset_cache(self):
        self._datasets = None

    def get_dataset_masks(self):
        vol_filter = filters.InRangeFilter(
            self.featureObj, 'volume', high=self._filter_max_monthly_volume,
            low=self.featureObj.filter_min_monthly_volume)

        dataset_masks = dict()
        intervals = self.get_intervals()
        for dataset_type, interval in intervals.items():
            dataset_mask = self.featureObj._init_data_panel(bool)
            dataset_mask[(self.featureObj.dates['m'] < interval[0])] = False
            dataset_mask[(self.featureObj.dates['m'] > interval[1])] = False
            dataset_masks[dataset_type] = self.featureObj.good_mask & \
                                               dataset_mask & \
                                               vol_filter.get_mask()
        return dataset_masks

    def _get_dataset_features_lists(self):
        dataset_masks = self.get_dataset_masks()
        dataset_features_lists = {k: [] for k in dataset_masks.keys()}
        for feature_name in self.feature_names:
            feature_arr = self.features_dict[feature_name]
            for dataset_type, mask in dataset_masks.items():
                dataset_features_lists[dataset_type].append(feature_arr[mask])
        return dataset_features_lists

    @property
    def groups(self):        
        # Get the # of samples in each group (aka the # of stocks for each date)
        groups = dict()
        for dataset_type, X_ds in self.X.items():
            ym_vals = self.year_month[dataset_type]
            assert np.all(np.sort(ym_vals) == ym_vals), 'Samples should be sorted by year-month'
            df_ym = pd.Series(ym_vals, name='year_month').reset_index()
            groups[dataset_type] = df_ym.groupby('year_month').count().sort_index().to_numpy()
        return groups
    
    def get_eval_set(self):
        eval_set = dict(bull=[], bear=[], reg=[], pct=[])
        year_month_vals = self.year_month['validation']
        for ym in sorted(set(year_month_vals)):
            idx_ym = (ym == year_month_vals)
            eval_set[BULL].append((self.X['validation'][idx_ym,:], self.y_class_norm['validation'][idx_ym]))
            eval_set[BEAR].append((self.X['validation'][idx_ym,:], 
                                   self.featureObj.n_buckets - 1 - self.y_class_norm['validation'][idx_ym]))
            eval_set['reg'].append(self.y_reg['validation'][idx_ym])
            eval_set['pct'].append(self.y_pct['validation'][idx_ym])
        return eval_set