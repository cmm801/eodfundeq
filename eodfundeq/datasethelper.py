import datetime
import numpy as np
import pandas as pd

from pypfopt import HRPOpt

from pyfintools.tools import optim

from eodfundeq.constants import START_DATE, DataSetTypes
from eodfundeq import filters
from eodfundeq import utils


BULL = 'bull'
BEAR = 'bear'


class DatasetHelper():
    def __init__(self, featureObj, features_dict, return_window=1, norm=True):
        self.featureObj = featureObj
        self.features_dict = features_dict
        self.return_window = return_window
        self._filter_max_monthly_volume = np.inf
        self.reset_cache()

        self.target_vol = 0.15
        self.exclude_nan = False
        self.norm = norm
        
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
            self._datasets = dict(X=dict(), year_month=dict(), symbol=dict(), symbol_idx=dict(),
                                  y_class=dict(), y_reg=dict(), y_pct=dict(),
                                  y_class_norm=dict(), y_reg_norm=dict(), y_pct_norm=dict(),
                                 )
            dataset_masks = self.get_dataset_masks()
            dataset_features_lists = self._get_dataset_features_lists()
            future_returns = self.featureObj.get_future_returns(self.return_window)
            if self.return_window == 1:
                volatility = self.featureObj.get_volatility(63, min_obs=60)
            elif self.return_window == 3:
                volatility = self.featureObj.get_volatility(126, min_obs=120)
            else:
                raise NotImplementedError('The volatility window corresponding to this return window has not been chosen.')

            n_symbols = len(self.featureObj.symbols)
            dates = self.featureObj.dates['m']
            ym_int = np.array([x.year * 100 + x.month for x in dates])
            ym_panel = np.tile(ym_int.reshape(-1, 1), n_symbols)
            symbol_index_panel = np.tile(np.arange(n_symbols).reshape(-1, 1), len(dates)).T
            symbol_panel = np.tile(self.featureObj.symbols.reshape(-1, 1), len(dates)).T

            for dataset_type, mask in dataset_masks.items():
                fut_rtns = future_returns.copy()
                fut_rtns[~mask] = np.nan
                bucketed_returns = self.featureObj.get_buckets(fut_rtns)
                pred_pct = utils.calc_feature_percentiles(bucketed_returns)
                pred_bkt = np.digitize(pred_pct, np.linspace(0, 1, self.featureObj.n_buckets+1)[:-1])
                y_0 = pred_bkt[mask] - 1

                fut_rtns_norm = fut_rtns / np.maximum(volatility, 0.03)
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
                self._datasets['symbol_idx'][dataset_type] = symbol_index_panel[mask][idx_no_nan]
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
    
    def get_eval_set(self, norm=True):
        eval_set = dict(bull=[], bear=[], reg=[], pct=[])
        year_month_vals = self.year_month['validation']
        for ym in sorted(set(year_month_vals)):
            idx_ym = (ym == year_month_vals)
            if self.norm:
                y_vals = self.y_class_norm['validation'][idx_ym]
            else:
                y_vals = self.y_class['validation'][idx_ym]

            eval_set[BULL].append((self.X['validation'][idx_ym,:], y_vals))
            eval_set[BEAR].append((self.X['validation'][idx_ym,:], self.featureObj.n_buckets - 1 - y_vals))
            eval_set['reg'].append(self.y_reg['validation'][idx_ym])
        return eval_set

    def get_selected_returns(self, model, dataset, n_stocks=100):
        year_month_vals = self.year_month[dataset]
        prices_ts = self.featureObj.daily_prices
        true_rtns_list = []
        selected_symbols_list = []
        period_ends = []
        sorted_year_months = sorted(set(year_month_vals))
        for ym in sorted_year_months:
            idx_ym = (ym == year_month_vals)
            y_score = model.predict(self.X[dataset][idx_ym,:])
            idx_score = np.argsort(y_score)[-n_stocks:][::-1]
            selected_symbols = self.symbol[dataset][idx_ym][idx_score]
            period_end = pd.Timestamp(datetime.datetime.strptime(str(ym), '%Y%m')) + pd.tseries.offsets.MonthEnd(0)
            period_ends.append(period_end)
            period_start = period_end - pd.Timedelta(91, unit='d')
            idx_period = (period_start <= prices_ts.index) & (prices_ts.index <= period_end)
            period_prices = prices_ts.loc[idx_period, selected_symbols]
            rtns = np.log(np.maximum(period_prices, self.featureObj.price_tol) / \
                        np.maximum(period_prices.shift(1), self.featureObj.price_tol))
            rtns = rtns.iloc[1:,:]
            assert np.isnan(rtns.values).sum(axis=0).max() < 10
            true_rtns = self.y_reg[dataset][idx_ym]
            true_rtns_list.append(true_rtns[idx_score])
            selected_symbols_list.append(selected_symbols)

        df_rtns = pd.DataFrame(true_rtns_list, index=pd.DatetimeIndex(period_ends))
        df_symbols = pd.DataFrame(selected_symbols_list, index=pd.DatetimeIndex(period_ends))
        return df_rtns, df_symbols

    def get_strategy_weights(self, model, dataset, n_stocks=100, weighting='equal'):
        _, df_symbols = self.get_selected_returns(model, dataset, n_stocks=n_stocks)
        prices_ts = self.featureObj.daily_prices    
        weights_list = []
        for period_end in df_symbols.index:
            if weighting == 'equal':
                weights = np.ones((n_stocks,)) / n_stocks
            else:
                period_symbols = df_symbols.loc[period_end].values
                period_start = period_end - pd.Timedelta(91, unit='d')
                idx_period = (period_start <= prices_ts.index) & (prices_ts.index <= period_end)
                period_prices = prices_ts.loc[idx_period, period_symbols]
                rtns = np.log(np.maximum(period_prices, self.featureObj.price_tol) / \
                            np.maximum(period_prices.shift(1), self.featureObj.price_tol))
                rtns = rtns.iloc[1:,:]
                assert np.isnan(rtns.values).sum(axis=0).max() < 10
                rtns.fillna(0, inplace=True)
                if weighting == 'hrp':        
                    hrp = HRPOpt(rtns)
                    hrp.optimize()
                    weights = pd.Series(hrp.clean_weights())
                    assert np.isclose(weights.sum(), 1.0, atol=0.01), 'Weights must sum to 1.0'
                elif weighting == 'minvar':
                    weights, _ = optim.optimize_min_variance(np.cov(rtns.T), ub=0.05)
                    assert np.isclose(weights.sum(), 1.0, atol=0.01), 'Weights must sum to 1.0'                
                elif weighting == 'vol':
                    realized_vol = rtns.std(axis=0).values * np.sqrt(252)
                    weights = 1/n_stocks * np.minimum(self.target_vol / realized_vol, 1.2)
                elif weighting == 'erc':
                    asset_cov = rtns.cov() * 252
                    lb = 1/n_stocks * 0.5
                    ub = 1/n_stocks * 2.0
                    weights = optim.optimize_erc(asset_cov, vol_lb=self.target_vol, vol_ub=self.target_vol,
                        lb=lb, ub=ub, unit_constraint=True)
                else:
                    raise ValueError(f'Unsupported weighting scheme: {weighting}')

            weights_list.append(weights)
        return pd.DataFrame(np.vstack(weights_list), index=df_symbols.index)

    def get_performance(self, model, n_stocks=100, dataset='validation', weighting='equal'):
        df_rtns, _ = self.get_selected_returns(model, dataset, n_stocks=n_stocks)
        df_wts = self.get_strategy_weights(model, dataset, n_stocks=n_stocks, weighting=weighting)
        realized_rtns = []
        for period_end in df_rtns.index:
            weights = df_wts.loc[period_end].values
            true_rtns = df_rtns.loc[period_end].values
            realized_rtns.append((weights * true_rtns).sum())
        return pd.Series(realized_rtns, index=df_rtns.index)

    def get_ndcg(self, model, n_stocks=100, dataset='validation'):
        year_month_vals = self.year_month[dataset]
        prices_ts = self.featureObj.daily_prices
        model_period_ndcg = []    
        sorted_year_months = sorted(set(year_month_vals))
        for ym in sorted_year_months:
            idx_ym = (ym == year_month_vals)
            y_score = model.predict(self.X[dataset][idx_ym,:])
            if self.norm:
                true_bkt = self.y_class_norm[dataset][idx_ym]
            else:
                true_bkt = self.y_class[dataset][idx_ym]

            ndcg_val = utils.calc_ndcg(true_bkt, y_score, k=n_stocks, form='exp')
            assert ndcg_val > 0
            model_period_ndcg.append(ndcg_val)
        return np.array(model_period_ndcg)