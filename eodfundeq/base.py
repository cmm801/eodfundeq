"""Define methods for analyzing fundamental data from eodhistoricaldata.com.

This file contains the core logic used for the analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats
import warnings

from typing import Optional

import pyfintools.tools.freq
from pyfintools.tools import tradingutils

from eodfundeq import utils
from eodfundeq.constants import ReturnTypes, DataSetTypes, FinancialStatementTypes, \
    FundamentalRatios, FundamentalRatioInputs, FINANCIAL_DATA_TYPE_MAP
from eodhistdata import EODHelper, FundamentalEquityData


class StockFeatureAnalyzer(object):
    periods_per_year = 12

    def __init__(self, api_token, base_path, start, end='', n_val_periods=24, n_test_periods=24, 
                 stale_days=120, dataset_type=DataSetTypes.TRAIN.value):
        self.eod_helper = EODHelper(api_token=api_token, base_path=base_path)

        self._dataset_mask = None
        self.n_val_periods = n_val_periods        
        self.n_test_periods = n_test_periods
        self.stale_days = stale_days

        self.start = pd.Timestamp(start)
        if not end:
            self.end = pd.Timestamp.now().round('d') - pd.tseries.offsets.MonthEnd(1)
        else:
            self.end = pd.Timestamp(end)
            
        self.start_str = self.start.strftime('%Y-%m-%d')
        self.end_str = self.end.strftime('%Y-%m-%d')

        self.dates = pd.date_range(self.start, self.end, freq='M')
        self.symbols = self.eod_helper.get_non_excluded_exchange_symbols('US')[:500]
        self._time_series = dict()
        self._market_cap = None
        
        # initialize hash table to look up date locations
        self._date_map_pd = pd.Series([], dtype=int)
        self._date_map_str = pd.Series([], dtype=int)
        for j, dt in enumerate(self.dates):
            self._date_map_pd[dt] = j
            self._date_map_str[dt.strftime('%Y-%m-%d')] = j
            
        # Initialize financial statement data types
        self.fin_data_types = FINANCIAL_DATA_TYPE_MAP.copy()
        
        # Initialize all indices to True
        self.good_index = self._create_data_panel(dtype=bool)

        # Specify which data set type we want to work with (train, validation, test)
        self.dataset_type = dataset_type
        
        # Other parameters
        self.price_tol = 1e-4           # Used to make sure we don't divide by 0
        self.n_buckets = 5              # How many buckets to use for metric sort
        self.fundamental_data_delay = 1 # Months to delay use of fundamental data, to account for
                                        # the fact that it is not immediately available for trading. 
        # Excludes dates/metrics with few observations
        self.filter_min_obs = int(self.n_symbols / (self.n_dates / 12 / 5) / self.n_buckets / 3)
        self.filter_min_price = 1       # Excludes stocks with too low of a price
        self.filter_min_monthly_volume = 21 * 5000  # Exclude stocks with low trading volume
        self.filter_max_return = 100  # Excludes return outliers from sample

    def _create_data_rows(self, n_rows, dtype=np.float32):
        return np.ones((n_rows, self.n_symbols), dtype=dtype)

    def _create_data_panel(self, dtype=np.float32):
        return self._create_data_rows(self.n_dates, dtype=dtype)

    def _get_loc(self, date):
        if isinstance(date, str):
            try:
                return self._date_map_str[date]
            except KeyError:
                return self._date_map_pd[pd.Timestamp(date) + pd.tseries.offsets.MonthEnd(0)]
        else:
            try:
                return self._date_map_pd[date]
            except KeyError:
                return self._date_map_pd[pd.Timestamp(date) + pd.tseries.offsets.MonthEnd(0)]

    def load_time_series(self):
        data_cols = {'close': np.nan, 'adjusted_close': np.nan, 'volume': 0.0}
        for col, default_val in data_cols.items():
            self._time_series[col] = default_val * self._create_data_panel(dtype=np.float32)

        for idx_symbol, symbol in enumerate(self.symbols):
            raw_ts = self.eod_helper.get_historical_data(
                symbol, start=self.start, stale_days=self.stale_days)
            if not raw_ts.shape[0]:
                continue

            monthly_ts = tradingutils.downsample(raw_ts, frequency='M')
            monthly_ts = monthly_ts.loc[(self.start <= monthly_ts.index) &
                                        (monthly_ts.index <= self.end)]
            if monthly_ts.shape[0] < 2:
                continue
            
            idx_date = self._get_loc(monthly_ts.index[0])
            if idx_date < 0:
                raise ValueError(f'Missing date: {monthly_ts.index[0]}')
            L = monthly_ts.shape[0]
            for col in data_cols.keys():
                self._time_series[col][idx_date:idx_date+L, idx_symbol] = monthly_ts[col].values
        
        self.good_index = \
            self.good_index & \
            ~np.isnan(self.close) & \
            (self.close > self.filter_min_price) & \
            ~np.isnan(self.adjusted_close) & \
            (self.adjusted_close > self.filter_min_price) & \
            ~np.isnan(self.volume) & \
            (self.volume > self.filter_min_monthly_volume)

        # Remove all series completely if there is a suspicious monthly return
        monthly_returns = self.get_future_returns(1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')        
            idx_high_return = np.nanmax(np.abs(monthly_returns), axis=0) > self.filter_max_return
        self.good_index[:, idx_high_return] = False
        
        # Fill non-trailing NaNs in close and adjusted_close time series
        self._time_series['close'] = utils.ffill(self._time_series['close'])
        self._time_series['adjusted_close'] = utils.ffill(self._time_series['adjusted_close'])

    @property
    def n_dates(self):
        return self.dates.size
    
    @property
    def n_symbols(self):
        return len(self.symbols)

    @property
    def dataset_type(self):
        return self._dataset_type
    
    @dataset_type.setter
    def dataset_type(self, dst):
        self._dataset_mask = None
        self._dataset_type = dst

    @property
    def dataset_mask(self):
        if self._dataset_mask is None:
            self._dataset_mask = self._create_data_panel(bool)
            if self.dataset_type == DataSetTypes.TRAIN.value:
                N = self.n_test_periods + self.n_val_periods
                self._dataset_mask[-N:, :] = False
            elif self.dataset_type == DataSetTypes.VALIDATION.value:
                N = self.n_test_periods
                self._dataset_mask[-N:, :] = False    
            elif self.dataset_type != DataSetTypes.TEST.value:
                raise ValueError(f'Unsupported dataset type: {self.dataset_type}')
                
        return self._dataset_mask
        
    @property
    def adjusted_close(self):
        if not len(self._time_series):
            self.load_time_series()
        return self._time_series['adjusted_close']
        
    @property
    def close(self):
        if not len(self._time_series):
            self.load_time_series()
        return self._time_series['close']
        
    @property
    def volume(self):
        if not len(self._time_series):
            self.load_time_series()
        return self._time_series['volume']

    @property
    def market_cap(self):
        if self._market_cap is None:
            self._market_cap = np.nan * self._create_data_panel()
            for idx_symbol, symbol in enumerate(self.symbols):
                raw_ts = self.eod_helper.get_market_cap(
                    symbol, start=self.start, stale_days=self.stale_days)
                if not raw_ts.shape[0]:
                    continue

                monthly_ts = raw_ts.resample('M').last()
                monthly_ts = monthly_ts.loc[(self.start <= monthly_ts.index) &
                                            (monthly_ts.index <= self.end)]
                if monthly_ts.shape[0] < 2:
                    continue

                idx_date = self._get_loc(monthly_ts.index[0])
                if idx_date < 0:
                    raise ValueError(f'Missing date: {monthly_ts.index[0]}')
                L = monthly_ts.shape[0]
                self._market_cap[idx_date:idx_date+L, idx_symbol] = monthly_ts.values.reshape(-1)

        return self._market_cap
    
    def get_future_returns(self, window, return_type=ReturnTypes.ARITHMETIC.value):
        levels = np.maximum(self.adjusted_close[window:, :], self.price_tol) /  \
            np.maximum(self.adjusted_close[:-window, :], self.price_tol)
        if return_type == ReturnTypes.ARITHMETIC.value:
            return np.vstack([-1 + levels, np.nan * self._create_data_rows(window)])
        elif return_type == ReturnTypes.LOG.value:
            return np.vstack([np.log(levels), np.nan * self._create_data_rows(window)])
        else:
            raise ValueError(f'Unsupported return type: {return_type}')
    
    def get_momentum(self, window, return_type=ReturnTypes.LOG.value, lag=0):
        if lag < 0:
            raise ValueError(f'Momentum lag must be positive. Found lag={lag}')
        if lag >= window:
            raise ValueError('Momentum lag must be less than the window length. '
                            f'Found window={window}, lag={lag}')

        eff_window = window - lag
        levels = np.maximum(self.adjusted_close[eff_window:, :], self.price_tol) /  \
            np.maximum(self.adjusted_close[:-eff_window, :], self.price_tol)
        if return_type == ReturnTypes.ARITHMETIC.value:
            mom_vals = -1 + levels
        elif return_type == ReturnTypes.LOG.value:
            mom_vals = np.log(levels)
        else:
            raise ValueError(f'Unsupported return type: {return_type}')
            
        return np.vstack([np.nan * self._create_data_rows(window),
                          mom_vals[:mom_vals.shape[0]-lag,:]])

    def get_financial_statement_data(self):
        # Initialize the financial data arrays
        fin_data = dict()
        for statement_type in FinancialStatementTypes:
            for fin_data_type in self.fin_data_types[statement_type.value]:
                fin_data[fin_data_type] = np.nan * self._create_data_panel()

        for idx_symbol, symbol in enumerate(self.symbols):
            fund_data = self.eod_helper.get_fundamental_equity(symbol)
            for statement_type in FinancialStatementTypes:
                statement = fund_data['Financials'][statement_type.value]['quarterly']

                for dt, period_data in statement.items():
                    if dt < self.start_str or dt > self.end_str:
                        continue

                    idx_dt = self._get_loc(dt) + self.fundamental_data_delay
                    if idx_dt == self.n_dates:
                        continue

                    for fin_data_type in self.fin_data_types[statement_type.value]:
                        fin_data[fin_data_type][idx_dt, idx_symbol] = period_data[fin_data_type]
        return fin_data

    def get_income_statement_data(self, fin_data_types=None):
        return self.get_financial_statement_data(
            statement_type=FinancialStatementTypes.INCOME_STATEMENT.value,
            fin_data_types=fin_data_types)

    def get_balance_sheet_data(self, fin_data_types=None):
        return self.get_financial_statement_data(
            statement_type=FinancialStatementTypes.BALANCE_SHEET.value,
            fin_data_types=fin_data_types)

    def get_cash_flow_data(self, fin_data_types=None):
        return self.get_financial_statement_data(
            statement_type=FinancialStatementTypes.CASH_FLOW.value,
            fin_data_types=fin_data_types)

    def _get_earnings(self):
        earnings = np.nan * self._create_data_panel()
        for idx_symbol, symbol in enumerate(self.symbols):
            fund_data = self.eod_helper.get_fundamental_equity(symbol)
            feq = FundamentalEquityData(fund_data)

            earnings_full = feq.earnings_ts
            if earnings_full.empty:
                continue

            earnings_ts = earnings_full.epsActual
            earnings_ts = earnings_ts.loc[(self.start <= earnings_ts.index) & \
                                          (self.end >= earnings_ts.index)]

            try:
                idx_series = self._date_map_pd[earnings_ts.index]
            except KeyError:
                idx_series = self._date_map_pd[earnings_ts.index + pd.tseries.offsets.MonthEnd(0)]

            idx = self.fundamental_data_delay + idx_series.values
            earnings[idx[idx < self.n_dates], idx_symbol] = earnings_ts.values[idx < self.n_dates]
        return earnings

    def get_earnings_yield(self, min_obs=4, fillna=False):
        earnings = self._get_earnings()
        annual_earnings = utils.rolling_sum(earnings, n_periods=12, min_obs=min_obs, fillna=fillna)
        return annual_earnings / (self.adjusted_close + self.price_tol)
    
    def bucket_results(self, metric_vals, return_window):
        return_vals = self.get_future_returns(return_window)
        assert metric_vals.shape == return_vals.shape, 'Shape of metric values must align with returns'

        bucketed_rtns = []
        bucketed_nobs = []
        high_vals = []
        low_vals = []
        for idx_date in range(self.n_dates):
            idx_keep = ~np.isnan(metric_vals[idx_date,:]) & \
                       ~np.isnan(return_vals[idx_date,:]) & \
                       self.good_index[idx_date,:] & \
                       self.dataset_mask[idx_date, :]

            metric_row = metric_vals[idx_date, idx_keep]
            return_row = return_vals[idx_date, idx_keep]

            if not metric_row.size:
                bucketed_nobs.append([0 for _ in range(self.n_buckets)])
                bucketed_rtns.append([np.nan for _ in range(self.n_buckets)])
                continue

            assert np.max(return_row) < 100
            bins = np.quantile(metric_row, np.linspace(0, 1, self.n_buckets+1))
            bins[-1] += .01  # Hack to avoid max value being assigned to its own bucket
            idx_bin = np.digitize(metric_row, bins, right=False)

            mean_rtns = []
            num_obs = []
            for bin_val in range(1, self.n_buckets+1):    
                num_obs.append(return_row[idx_bin == bin_val].size)
                if num_obs[-1] > 0:
                    mean_rtns.append(return_row[idx_bin == bin_val].mean())
                else:
                    mean_rtns.append(np.nan)

            bucketed_rtns.append(mean_rtns)
            bucketed_nobs.append(num_obs)
            low_vals.append(return_row[idx_bin == 1].reshape(-1))
            high_vals.append(return_row[idx_bin == self.n_buckets].reshape(-1))

        high = np.hstack(high_vals) if len(high_vals) else np.array([])
        low = np.hstack(low_vals) if len(low_vals) else np.array([])
        if high.size & low.size:
            tstat = scipy.stats.ttest_ind(high, low).statistic
        else:
            tstat = np.nan
        return np.vstack(bucketed_rtns), np.vstack(bucketed_nobs), tstat

    def get_performance_ts(self, metric_vals, return_window):
        period_rtns, num_obs, tstat = self.bucket_results(metric_vals, 
                                                          return_window=return_window)
        idx_keep_rows = np.min(num_obs, axis=1) >= self.filter_min_obs
        period_rtns = period_rtns[idx_keep_rows, :]
        good_dates = self.dates[idx_keep_rows]

        if period_rtns.shape[0] == 0:
            return pd.DataFrame(), pd.DataFrame(), np.nan

        # Convert to approximate monthly returns so we can compute hypothetical performance
        monthly_rtns = -1 + (1 + period_rtns) ** (1/return_window)
        perf_ts = pd.DataFrame(np.cumprod(1 + monthly_rtns, axis=0), index=good_dates)
        obs_ts = pd.DataFrame(num_obs[idx_keep_rows], index=good_dates)
        return perf_ts, obs_ts, tstat

    def get_bucketed_returns_summary(self, metric_vals, return_windows: list):
        if return_windows is None:
            return_windows = RETURN_WINDOWS

        results = dict()
        res_map = dict()
        ann_returns = dict()
        perf_ts = pd.DataFrame()  # Initialize this in case there is no data
        for window in return_windows:
            perf_ts, obs_ts, tstat = self.get_performance_ts(
                metric_vals, return_window=window)
            if not perf_ts.size:
                continue

            # Get the number of periods per year so we can annualize the cum. return
            n_years = perf_ts.shape[0] / 12
            bucket_means = (-1 + perf_ts ** (1/n_years)).tail(1).values[0]

            # Combine results into dict for output
            res_map[window] = dict(
                low=bucket_means[0],
                high=bucket_means[-1],
                alpha=bucket_means[-1] - bucket_means[0],
                tstat=tstat,
                n_obs=np.sum(obs_ts.values),
                min_obs=np.min(obs_ts.values),
                n_dates=perf_ts.shape[0])
            ann_returns[window] = bucket_means
        results['summary'] = pd.DataFrame(res_map).T
        results['ann_returns'] = pd.DataFrame(ann_returns).T
        return results
    
    def get_bucket_summary_for_momentum(self, 
                                        momentum_windows, 
                                        return_windows,
                                        lags=(1,)):
        lags = sorted(list(set(lags) | set([0])))
        results = dict()
        for momentum_window in momentum_windows:
            for lag in lags:
                if lag < momentum_window:
                    mom_ts = featureObj.get_momentum(momentum_window, lag=lag)
                    key = f'mom_{momentum_window}m{lag}'
                    results[key] = self.get_bucketed_returns_summary(
                        mom_ts, return_windows=return_windows)
        return results    
    
    def get_bucket_summary_for_valuation(self, return_windows, fillna=False):
        results = dict()
        earnings_yield = featureObj.get_earnings_yield(fillna=fillna)
        results['earnings_yield'] = self.get_bucketed_returns_summary(
            earnings_yield, return_windows=return_windows)
        return results
    
    def get_bucket_summary_for_fundamental_ratios(self, return_windows, fillna=False,
                                                  n_periods=None, min_obs=4):
        fin_data = self.get_financial_statement_data()
        
        results = dict()
        for ratio_type in FundamentalRatios:            
            ratio_vals = self.calculate_fundamental_ratios(ratio_type.value, 
                fin_data, n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            results[ratio_type.value] = self.get_bucketed_returns_summary(
                ratio_vals, return_windows=return_windows)
        return results

    def calculate_fundamental_ratios(self, ratio, fin_data, n_periods=None,
                                     min_obs=4, fillna=False):
        if n_periods is None:
            n_periods = self.periods_per_year

        eps = 1e-10
        if ratio == FundamentalRatios.ROA.value:
            net_income = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.NET_INCOME.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)

            total_assets = fin_data[FundamentalRatioInputs.TOTAL_ASSETS.value]
            avg_total_assets = np.vstack([
                np.nan * self._create_data_rows(n_periods),
                total_assets[:-n_periods,:] + total_assets[n_periods:,:]
            ])
            return net_income / (avg_total_assets + eps)
        elif ratio == FundamentalRatios.ROE.value:
            net_income = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.NET_INCOME.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)

            equity = fin_data[FundamentalRatioInputs.SHAREHOLDER_EQUITY.value]
            avg_equity = np.vstack([
                np.nan * self._create_data_rows(n_periods),
                equity[:-n_periods,:] + equity[n_periods:,:]
            ])
            return net_income / (avg_equity + eps)
        elif ratio == FundamentalRatios.ROIC.value:
            income_pre_tax = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.INCOME_BEFORE_TAX.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            tax = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.INCOME_TAX_EXPENSE.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            tax_rate = tax / (eps + income_pre_tax)

            ebit = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.EBIT.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            nopat = ebit * (1 - tax_rate)

            capital = fin_data[FundamentalRatioInputs.NET_INVESTED_CAPITAL.value]
            avg_capital = np.vstack([
                np.nan * self._create_data_rows(n_periods),
                capital[:-n_periods,:] + capital[n_periods:,:]
            ])
            return nopat / (avg_capital + eps)
        elif ratio == FundamentalRatios.GROSS_MARGIN.value:
            gross_profit = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.GROSS_PROFIT.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)        
            total_revenue = fin_data[FundamentalRatioInputs.TOTAL_REVENUE.value]
            return gross_profit / (total_revenue + eps)
        elif ratio == FundamentalRatios.OPERATING_MARGIN.value:
            ebit = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.EBIT.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)        
            total_revenue = fin_data[FundamentalRatioInputs.TOTAL_REVENUE.value]
            return ebit / (total_revenue + eps)
        elif ratio == FundamentalRatios.NET_MARGIN.value:
            net_income = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.NET_INCOME.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            total_revenue = fin_data[FundamentalRatioInputs.TOTAL_REVENUE.value]
            return net_income / (total_revenue + eps)
        elif ratio == FundamentalRatios.CASH_FLOW_MARGIN.value:
            free_cash_flow = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.FREE_CASH_FLOW.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)        
            total_revenue = fin_data[FundamentalRatioInputs.TOTAL_REVENUE.value]
            return free_cash_flow / (total_revenue + eps)
        elif ratio == FundamentalRatios.CASH_FLOW_TO_NET_INCOME.value:
            free_cash_flow = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.FREE_CASH_FLOW.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)        
            net_income = utils.rolling_sum(
                fin_data[FundamentalRatioInputs.NET_INCOME.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return free_cash_flow / (net_income + eps)
        elif ratio == FundamentalRatios.EARNINGS_YIELD.value:
            return self.get_earnings_yield(min_obs=min_obs, fillna=fillna)
        else:
            raise ValueError(f'Unsupported fundamental ratio: {ratio}')