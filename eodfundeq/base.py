"""Define methods for analyzing fundamental data from eodhistoricaldata.com.

This file contains the core logic used for the analysis.
"""

import numpy as np
import pandas as pd
import pandas_market_calendars
import scipy.stats

from collections.abc import Iterable
from enum import Enum
from typing import Optional, Union

import pyfintools.tools.freq
from pyfintools.tools import tradingutils

from eodfundeq import utils
from eodfundeq.constants import ReturnTypes, DataSetTypes, FundamentalRatios, FUNDAMENTAL_RATIO_INPUTS
from eodfundeq.filters import EqualFilter, InRangeFilter, EntireColumnInRangeFilter, IsNotNAFilter

from eodhistdata import EODHelper, FundamentalEquitySnapshot, FundamentalEquityTS
from eodhistdata.constants import FundamentalDataTypes, TimeSeriesNames

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252


class TSNames(Enum):
    ADJUSTED_CLOSE = TimeSeriesNames.ADJUSTED_CLOSE.value
    CLOSE = TimeSeriesNames.CLOSE.value
    DAILY_PRICES = 'daily_prices'
    MARKET_CAP = 'market_cap'
    MONTHLY_RETURNS = 'monthly_returns'    
    VOLUME = TimeSeriesNames.VOLUME.value


TIME_SERIES_PANEL_INFO = {
    TSNames.ADJUSTED_CLOSE.value: dict(dtype=np.float32, frequency='m'),
    TSNames.CLOSE.value: dict(dtype=np.float32, frequency='m'),
    TSNames.DAILY_PRICES.value: dict(dtype=np.float32, frequency='b'),
    TSNames.MARKET_CAP.value: dict(dtype=np.float32, frequency='m'),    
    TSNames.MONTHLY_RETURNS.value: dict(dtype=np.float32, frequency='m'),
    TSNames.VOLUME.value: dict(dtype=np.float32, frequency='m'),
}

class StockFeatureAnalyzer(object):
    periods_per_year = 12

    def __init__(self, api_token, base_path, start, end='', symbols=None, 
                 n_val_periods=24, n_test_periods=24, stale_days=240, clip=None, 
                 drop_empty_ts=True):
        self.eod_helper = EODHelper(api_token=api_token, base_path=base_path)
        self.n_val_periods = n_val_periods        
        self.n_test_periods = n_test_periods
        self.stale_days = stale_days
        self.clip = clip if clip is not None else (-np.inf, np.inf)
        self.drop_empty_ts = drop_empty_ts

        self.start = pd.Timestamp(start)
        if not end:
            self.end = pd.Timestamp.now().round('d') - pd.tseries.offsets.MonthEnd(1)
        else:
            self.end = pd.Timestamp(end)

        self.start_str = self.start.strftime('%Y-%m-%d')
        self.end_str = self.end.strftime('%Y-%m-%d')

        self.frequencies = ('b', 'm')
        self.exchange_id = 'US'
        if self.exchange_id == 'US':
            market_calendar = pandas_market_calendars.get_calendar('NYSE')
        else:
            raise NotImplementedError('Only implemented for US exchanges.')
        self.dates = {'m': pd.date_range(self.start, self.end, freq='m')}            
        business_days = market_calendar.valid_days(self.start, self.end)
        self.dates['b'] = pd.DatetimeIndex([x.tz_convert(None) for x in business_days])
        
        # initialize hash table to look up date locations
        self._date_map = {f: pd.Series({self.dates[f][j]: j for j in range(len(self.dates[f]))},
             dtype=int) for f in self.frequencies}

        if symbols is None:
            self.symbols = np.array(self.eod_helper.get_non_excluded_exchange_symbols('US'))
        else:
            self.symbols = np.array(symbols)

        # Initialize the time series panels
        self._time_series = dict()

        # Initialize cached time series data
        self._cta_momentum_signals = None
            
        # Initialize financial statement data types
        self.fin_data_types = sorted(FUNDAMENTAL_RATIO_INPUTS)
        
        # Initialize mask that will tell us if data points are valid
        self._good_mask = None

        # Initialize container for caching calculated volatility
        self._volatility = dict()

        # Initialize container for caching calculated fundamental ratios
        self._fundamental_ratios = None        
        
        # Other parameters
        self.price_tol = 1e-4           # Used to make sure we don't divide by 0
        self.n_buckets = 5              # How many buckets to use for metric sort
        self.fundamental_data_delay = 1 # Months to delay use of fundamental data, to account for
                                        # the fact that it is not immediately available for trading. 
        self.filter_min_obs = 10        # Excludes dates/metrics with few observations
        self.filter_min_price = 1       # Excludes stocks with too low of a price
        self.filter_min_monthly_volume = 21 * 10000  # Exclude stocks with low trading volume
        self.filter_max_return = 10     # Excludes return outliers from sample

        # Set default filters
        self.set_default_filters()

    def _init_data_rows(self, n_rows, dtype=np.float32):
        return np.ones((n_rows, self.symbols.size), dtype=dtype)

    def _init_data_panel(self, dtype=np.float32, frequency='m'):
        return self._init_data_rows(self.dates[frequency].size, dtype=dtype)

    def _init_pd_dataframe(self, frequency, dtype=np.float32):
        return pd.DataFrame(
            np.ones((self.dates[frequency].size, self.symbols.size), dtype=dtype),
            index=self.dates[frequency], columns=self.symbols)

    def _get_loc(self, date):
        try:
            return self._date_map[date]
        except KeyError:
            return self._date_map[pd.Timestamp(date) + pd.tseries.offsets.MonthEnd(0)]

    def load_ohlcv_data(self):
        ts_types = (TSNames.ADJUSTED_CLOSE, TSNames.CLOSE, TSNames.DAILY_PRICES,
                    TSNames.MONTHLY_RETURNS, TSNames.VOLUME)
        if np.all([t in self._time_series for t in ts_types]):
            return  # Time series are already loaded

        for name in ts_types:
            kwargs = TIME_SERIES_PANEL_INFO[name.value]
            default_val = 0.0 if name.value == TSNames.VOLUME.value else np.nan
            self._time_series[name.value] = default_val * self._init_pd_dataframe(**kwargs)

        for idx_symbol, symbol in enumerate(self.symbols):
            daily_ts = self.eod_helper.get_historical_data(
                symbol, start=self.start, stale_days=self.stale_days)
            if not daily_ts.shape[0]:
                continue

            daily_ts = daily_ts.loc[(self.start <= daily_ts.index) & \
                                    (daily_ts.index <= self.end)]

            # Clean data, removing bad prices
            daily_ts = self._clean_ohlcv_data(daily_ts)

            # Add daily prices (adjusted closing price) to the panel
            idx_shared = daily_ts.index.isin(self.dates['b'])
            self._time_series[TSNames.DAILY_PRICES.value].loc[daily_ts.index[idx_shared], symbol] = \
                    daily_ts.adjusted_close.values[idx_shared].astype(np.float32)

            # Downsample to monthly data (from daily)
            monthly_ts = tradingutils.downsample(daily_ts, frequency='M')
            if monthly_ts.shape[0] < 2:
                continue

            # Add each time series' data to the respective panel
            for ts_enum in (TSNames.ADJUSTED_CLOSE, TSNames.CLOSE, TSNames.VOLUME):
                idx_shared = monthly_ts.index.isin(self.dates['m'])
                self._time_series[ts_enum.value].loc[monthly_ts.index[idx_shared], symbol] = \
                    monthly_ts[ts_enum.value].values[idx_shared].astype(np.float32)

        # Also add monthly returns
        adj_close = self._time_series[TSNames.ADJUSTED_CLOSE.value]
        self._time_series[TSNames.MONTHLY_RETURNS.value] = \
            -1 + adj_close / np.maximum(adj_close.shift(1).values, self.price_tol)

        if self.drop_empty_ts:
            self._drop_empty_time_series()

    def _clean_ohlcv_data(self, daily_ts):
        daily_ts = daily_ts.query('volume > 0')
        spikes = True
        while spikes:
            prices = daily_ts.adjusted_close
            down_spike_mask = \
                (prices / np.maximum(prices.shift(1), self.price_tol) < 0.5) & \
                (prices.shift(-1) / np.maximum(prices, self.price_tol) > 1.9) & \
                ((prices.shift(1) > 1)) & \
                ((prices.shift(-1) > 1))

            up_spike_mask = \
                (prices / np.maximum(prices.shift(1), self.price_tol) > 1.9) & \
                (prices.shift(-1) / np.maximum(prices, self.price_tol) < 0.5) & \
                ((prices.shift(1) > 1)) & \
                ((prices.shift(-1) > 1))

            mask = down_spike_mask.values | up_spike_mask.values
            if not np.any(mask):
                spikes = False
            else:
                daily_ts = daily_ts.loc[~mask]
        return daily_ts

    def _drop_empty_time_series(self):
        valid_prices_per_symbol = (~np.isnan(self.daily_prices)).sum(axis=0).values
        idx_good_symbol = valid_prices_per_symbol >= TRADING_DAYS_PER_YEAR 
        self.symbols = self.symbols[idx_good_symbol]
        for k, v in self._time_series.items():
            self._time_series[k] = self._time_series[k].loc[:,self.symbols]

    def load_market_cap_data(self):
        mc_name = TSNames.MARKET_CAP.value
        kwargs = TIME_SERIES_PANEL_INFO[mc_name]
        if mc_name in self._time_series:
            return  # Time series already loaded
        self._time_series[mc_name] = np.nan * self._init_pd_dataframe(**kwargs)
        for idx_symbol, symbol in enumerate(self.symbols):
            raw_ts = self.eod_helper.get_market_cap(
                symbol, start=self.start, stale_days=self.stale_days)
            if not raw_ts.shape[0]:
                continue
            monthly_ts = raw_ts.resample('M').last()
            monthly_ts = monthly_ts.loc[(self.start <= monthly_ts.index) &
                                        (monthly_ts.index <= self.end)]
            if monthly_ts.shape[0] > 2:
                idx_shared = monthly_ts.index.isin(self.dates['m'])
                self._time_series[mc_name].loc[monthly_ts.index[idx_shared], symbol] = \
                    monthly_ts.values[idx_shared].flatten().astype(np.float32)

    def set_default_filters(self):
        self._good_mask = None
        self._filters = [
            IsNotNAFilter(self, TSNames.CLOSE.value),
            IsNotNAFilter(self, TSNames.ADJUSTED_CLOSE.value),
            IsNotNAFilter(self, TSNames.ADJUSTED_CLOSE.value,
                          property_func=lambda x: x.rolling(12).mean()),
            InRangeFilter(self, TSNames.DAILY_PRICES.value, high=3, high_inc=True,
                          property_func=lambda x: np.isnan(x).rolling(63).sum().resample('M').last()),
            IsNotNAFilter(self, TSNames.VOLUME.value),
            InRangeFilter(self, TSNames.CLOSE.value,
                          low=self.filter_min_price),
            InRangeFilter(self, TSNames.ADJUSTED_CLOSE.value,
                          low=self.filter_min_price),
            InRangeFilter(self, TSNames.VOLUME.value,
                          property_func=lambda x: x.rolling(12).quantile(0.01, interpolation='lower'),
                          low=self.filter_min_monthly_volume),
            EntireColumnInRangeFilter(self, TSNames.MONTHLY_RETURNS.value,
                                      high=self.filter_max_return)
        ]

    def reset_default_filters(self):
        self.set_default_filters()

    def add_filter(self, f):
        self._good_mask = None
        self._filters.append(f)

    @property
    def filters(self):
        return self._filters

    @property
    def good_mask(self):
        if self._good_mask is None:
            self._good_mask = self._init_pd_dataframe(frequency='m', dtype=bool)
            for f in self.filters:
                self._good_mask &= f.get_mask()

        return self._good_mask.values

    @property
    def adjusted_close(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.ADJUSTED_CLOSE.value]
        
    @property
    def close(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.CLOSE.value]

    @property
    def daily_prices(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.DAILY_PRICES.value]

    @property
    def monthly_returns(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.MONTHLY_RETURNS.value]

    @property
    def volume(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.VOLUME.value]

    @property
    def market_cap(self):
        if TSNames.MARKET_CAP.value not in self._time_series:
            self.load_market_cap_data()
        return self._time_series[TSNames.MARKET_CAP.value]
    
    def get_future_returns(self, window, return_type=ReturnTypes.ARITHMETIC.value):
        levels = np.maximum(self.adjusted_close.shift(-window).values, self.price_tol) /  \
                 np.maximum(self.adjusted_close.values, self.price_tol)
        if return_type == ReturnTypes.ARITHMETIC.value:
            return -1 + levels
        elif return_type == ReturnTypes.LOG.value:
            return np.log(levels)
        else:
            raise ValueError(f'Unsupported return type: {return_type}')

    def get_momentum(self, window, return_type=ReturnTypes.LOG.value, lag=0):
        if lag < 0:
            raise ValueError(f'Momentum lag must be positive. Found lag={lag}')
        if lag >= window:
            raise ValueError('Momentum lag must be less than the window length. '
                            f'Found window={window}, lag={lag}')
        levels = np.maximum(self.adjusted_close.shift(lag).values, self.price_tol) /  \
                 np.maximum(self.adjusted_close.shift(window).values, self.price_tol)
        if return_type == ReturnTypes.ARITHMETIC.value:
            return -1 + levels
        elif return_type == ReturnTypes.LOG.value:
            return np.log(levels)
        else:
            raise ValueError(f'Unsupported return type: {return_type}')

    def get_volatility(self, window: int, return_type=ReturnTypes.LOG.value, min_obs=None, fillna=True):
        daily_prices = self.daily_prices
        daily_log_rtns = np.log(np.maximum(daily_prices, self.price_tol) / \
                                np.maximum(daily_prices.shift(1).values, self.price_tol))
        daily_log_rtns.values[np.isclose(self.daily_prices, 0)] = np.nan
        rolling_daily_vol = daily_log_rtns.rolling(window, axis=0, min_periods=min_obs).std() * \
                            np.sqrt(TRADING_DAYS_PER_YEAR)
        monthly_vol = rolling_daily_vol.resample('M').last()
        if np.all(monthly_vol.index.values == self.dates['m']):
            return monthly_vol.values
        else:
            raise ValueError('Unexpected dates found in rolling vol.')

    def get_financial_statement_input_data(self, frequency='m'):
        """Gather all required input data from financial statements.
        
        This method returns a dict for each fundamental equity input type.
        Each dict has as its value a panel of float data, with one column for 
        each ticker symbol and one row for each date between start/end dates.
        The method loops through each ticker symbol, downloads the fundamental 
        time series data, and then puts that data into the respective time 
        series panel.
        """
        # Initialize the financial data arrays
        fin_data = dict()
        for fin_data_type in self.fin_data_types:
            fin_data[fin_data_type] = np.nan * self._init_data_panel()

        # For each symbol in our universe, get time series for all required input data types
        n_dates = self.dates[frequency].size        
        for idx_symbol, symbol in enumerate(self.symbols):
            fund_data = self.eod_helper.get_fundamental_equity(symbol, stale_days=self.stale_days)
            feq_ts_obj = FundamentalEquityTS(fund_data)
            full_ts = feq_ts_obj.get_time_series(self.fin_data_types)
            if not full_ts.size:
                continue

            for fin_data_type in self.fin_data_types:
                if fin_data_type not in full_ts:
                    continue

                ts = full_ts[fin_data_type]  # Keep just the column we want

                # Make sure all dates are between the start/end dates
                ts = ts.loc[(self.start_str <= ts.index) & (ts.index <= self.end_str)]

                # Get the row locations in the final time series panel for each observation
                try:
                    idx_series = self._date_map[ts.index]
                except KeyError:
                    # If one of the dates is not in the target index, try moving dates to month-end
                    dates_pd = pd.DatetimeIndex(ts.index) + pd.tseries.offsets.MonthEnd(0)
                    idx_series = self._date_map[dates_pd]

                idx = self.fundamental_data_delay + idx_series.values
                fin_data[fin_data_type][idx[idx < n_dates], idx_symbol] = ts.values[idx < n_dates]
        return fin_data

    def bucket_results(self, metric_vals, return_window, clip=None, frequency='m'):
        if clip is None:
            clip = self.clip
        return_vals = np.clip(self.get_future_returns(return_window), *clip)
        assert metric_vals.shape == return_vals.shape, 'Shape of metric values must align with returns'

        bucketed_rtns = []
        bucketed_nobs = []
        high_vals = []
        low_vals = []

        all_years = sorted(set([x.year for x in self.dates]))
        annual_high_vals = {y: [] for y in all_years}
        annual_low_vals = {y: [] for y in all_years}
        for idx_date, date in enumerate(self.dates[frequency]):
            idx_keep = ~np.isnan(metric_vals[idx_date,:]) & \
                       ~np.isnan(return_vals[idx_date,:]) & \
                       self.good_mask[idx_date,:]

            metric_row = metric_vals[idx_date, idx_keep]
            return_row = return_vals[idx_date, idx_keep]

            if not metric_row.size:
                bucketed_nobs.append([0 for _ in range(self.n_buckets)])
                bucketed_rtns.append([np.nan for _ in range(self.n_buckets)])
                continue

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
            period_low_vals = return_row[idx_bin == 1].reshape(-1)
            period_high_vals = return_row[idx_bin == self.n_buckets].reshape(-1)
            annual_high_vals[date.year].extend(period_high_vals)
            annual_low_vals[date.year].extend(period_low_vals)

        # First flatten the annual high/low values into a single array to calculate full t-stat
        high_vals = [x for sublist in list(annual_high_vals.values()) for x in sublist]
        low_vals = [x for sublist in list(annual_low_vals.values()) for x in sublist]
        high = np.hstack(high_vals) if len(high_vals) else np.array([])
        low = np.hstack(low_vals) if len(low_vals) else np.array([])
        if high.size & low.size:
            overall_tstat = scipy.stats.ttest_ind(high, low).statistic
        else:
            overall_tstat = np.nan

        # Calculate t-stat on a each of the annual sets of high/low values
        annual_tstat_vals = []
        for year in all_years:
            if len(annual_low_vals[year]) > 3:
                t = scipy.stats.ttest_ind(annual_high_vals[year], annual_low_vals[year]).statistic
                annual_tstat_vals.append(t)
            else:
                annual_tstat_vals.append(np.nan)
        ann_tstat_ts = pd.Series(annual_tstat_vals, 
                             index=pd.DatetimeIndex([f'{y}-12-31' for y in all_years]))

        return np.vstack(bucketed_rtns), np.vstack(bucketed_nobs), ann_tstat_ts, overall_tstat

    def get_buckets(self, metric_vals, frequency='m'):
        buckets = np.nan * np.ones_like(metric_vals, dtype=np.int32)
        for idx_date, date in enumerate(self.dates[frequency]):
            idx_keep = ~np.isnan(metric_vals[idx_date,:]) & \
                       self.good_mask[idx_date,:]
            metric_row = metric_vals[idx_date, idx_keep]
            if not metric_row.size:
                continue
            bins = np.quantile(metric_row, np.linspace(0, 1, self.n_buckets+1))
            bins[-1] += .01  # Hack to avoid max value being assigned to its own bucket
            buckets[idx_date, idx_keep] = np.digitize(metric_row, bins, right=False)
        return buckets

    def get_performance_ts(self, metric_vals, return_window, clip=None):
        period_rtns, num_obs, ann_tstat_ts, overall_tstat = self.bucket_results(
            metric_vals, return_window=return_window, clip=clip)
        idx_keep_rows = np.min(num_obs, axis=1) >= self.filter_min_obs
        period_rtns = period_rtns[idx_keep_rows, :]
        good_dates = self.dates[idx_keep_rows]

        if period_rtns.shape[0] == 0:
            return pd.DataFrame(), pd.DataFrame(), ann_tstat_ts, np.nan

        # Convert to approximate monthly returns so we can compute hypothetical performance
        monthly_rtns = -1 + (1 + period_rtns) ** (1/return_window)
        perf_ts = pd.DataFrame(np.cumprod(1 + monthly_rtns, axis=0), index=good_dates)
        obs_ts = pd.DataFrame(num_obs[idx_keep_rows], index=good_dates)
        return perf_ts, obs_ts, ann_tstat_ts, overall_tstat

    def get_bucketed_returns_summary(self, metric_vals, return_windows: list, 
                                    clip: Optional[tuple] = None):
        if return_windows is None:
            return_windows = RETURN_WINDOWS

        results = dict()
        res_map = dict()
        ann_returns = dict()
        perf_ts = pd.DataFrame()  # Initialize this in case there is no data
        for window in return_windows:
            perf_ts, obs_ts, ann_tstat_ts, overall_tstat = self.get_performance_ts(
                metric_vals, return_window=window, clip=clip)
            if not perf_ts.size:
                continue

            # Get the number of periods per year so we can annualize the cum. return
            n_years = perf_ts.shape[0] / self.periods_per_year
            bucket_means = (-1 + perf_ts ** (1/n_years)).tail(1).values[0]

            # Combine results into dict for output
            res_map[window] = dict(
                low=bucket_means[0],
                high=bucket_means[-1],
                alpha=bucket_means[-1] - bucket_means[0],
                overall_tstat=overall_tstat,
                tstat_25=ann_tstat_ts.quantile(0.25),
                tstat_50=ann_tstat_ts.quantile(0.50),
                tstat_75=ann_tstat_ts.quantile(0.75),
                n_obs=np.sum(obs_ts.values),
                min_obs=np.min(obs_ts.values),
                n_dates=perf_ts.shape[0])
            ann_returns[window] = bucket_means
        results['summary'] = pd.DataFrame(res_map).T
        results['ann_returns'] = pd.DataFrame(ann_returns).T
        results['perf_ts'] = perf_ts
        results['tstat_ts'] = ann_tstat_ts
        return results

    def get_bucket_summary_for_momentum(self, 
                                        momentum_windows, 
                                        return_windows,
                                        lags=(),
                                        clip=None):
        lags = sorted(list(set(lags) | set([0])))
        results = dict()
        for momentum_window in momentum_windows:
            for lag in lags:
                if lag < momentum_window:
                    mom_ts = self.get_momentum(momentum_window, lag=lag)
                    key = f'mom_{momentum_window}m'
                    if lag > 0:
                        key += str(lag)
                    results[key] = self.get_bucketed_returns_summary(
                        mom_ts, return_windows=return_windows, clip=clip)
        return results

    def calc_all_fundamental_ratios(self, fillna=False, n_periods=None, min_obs=4):
        if self._fundamental_ratios is None:
            self._fundamental_ratios = dict()
            fin_data = self.get_financial_statement_input_data()            
            for ratio_type in FundamentalRatios:
                self._fundamental_ratios[ratio_type.value] = self.calculate_fundamental_ratio(
                    ratio_type.value, fin_data, n_periods=n_periods, min_obs=min_obs, fillna=fillna)
        return self._fundamental_ratios

    def get_bucket_summary_for_fundamental_ratios(self, return_windows, fillna=False,
                                                  n_periods=None, min_obs=4, clip=None):
        if self._fundamental_ratios is None:
            self.calc_all_fundamental_ratios(fillna=fillna,
                n_periods=n_periods, min_obs=min_obs)
            self._fundamental_ratios = dict()
            fin_data = self.get_financial_statement_input_data()            
            for ratio_type in FundamentalRatios:
                self._fundamental_ratios[ratio_type.value] = self.calculate_fundamental_ratio(
                    ratio_type.value, fin_data, n_periods=n_periods, min_obs=min_obs, fillna=fillna)

        results = dict()
        for ratio_type in FundamentalRatios:
            results[ratio_type.value] = self.get_bucketed_returns_summary(
                self._fundamental_ratios[ratio_type.value], 
                return_windows=return_windows,
                clip=clip)
        return results

    def calculate_fundamental_ratio(self, ratio_name, fin_data, n_periods=None,
                                     min_obs=4, fillna=False):
        if n_periods is None:
            n_periods = self.periods_per_year

        eps = 1e-10
        if ratio_name == FundamentalRatios.ROA.value:
            net_income = utils.rolling_sum(
                fin_data[FundamentalDataTypes.netIncome.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)

            total_assets = fin_data[FundamentalDataTypes.totalAssets.value]
            begin_period_assets = np.vstack([np.nan * self._init_data_rows(n_periods), 
                                            total_assets[:-n_periods,:]])
            return net_income / (begin_period_assets + eps)
        elif ratio_name == FundamentalRatios.ROE.value:
            net_income = utils.rolling_sum(
                fin_data[FundamentalDataTypes.netIncome.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            equity = fin_data[FundamentalDataTypes.totalStockholderEquity.value]
            begin_period_equity = np.vstack([np.nan * self._init_data_rows(n_periods),
                                            equity[:-n_periods,:]])
            return net_income / (begin_period_equity + eps)
        elif ratio_name == FundamentalRatios.ROIC.value:
            income_pre_tax = utils.rolling_sum(
                fin_data[FundamentalDataTypes.incomeBeforeTax.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            tax = utils.rolling_sum(
                fin_data[FundamentalDataTypes.incomeTaxExpense.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            tax_rate = tax / (eps + income_pre_tax)
            ebit = utils.rolling_sum(
                fin_data[FundamentalDataTypes.ebit.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            nopat = ebit * (1 - tax_rate)
            capital = fin_data[FundamentalDataTypes.netInvestedCapital.value]
            begin_period_capital = np.vstack([np.nan * self._init_data_rows(n_periods),
                                             capital[:-n_periods,:]])
            return nopat / (begin_period_capital + eps)
        elif ratio_name == FundamentalRatios.GROSS_MARGIN.value:
            gross_profit = utils.rolling_sum(
                fin_data[FundamentalDataTypes.grossProfit.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            total_revenue = utils.rolling_sum(
                fin_data[FundamentalDataTypes.totalRevenue.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return gross_profit / (total_revenue + eps)
        elif ratio_name == FundamentalRatios.OPERATING_MARGIN.value:
            ebit = utils.rolling_sum(
                fin_data[FundamentalDataTypes.ebit.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)        
            total_revenue = utils.rolling_sum(
                fin_data[FundamentalDataTypes.totalRevenue.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return ebit / (total_revenue + eps)
        elif ratio_name == FundamentalRatios.NET_MARGIN.value:
            net_income = utils.rolling_sum(
                fin_data[FundamentalDataTypes.netIncome.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            total_revenue = utils.rolling_sum(
                fin_data[FundamentalDataTypes.totalRevenue.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return net_income / (total_revenue + eps)
        elif ratio_name == FundamentalRatios.FREE_CASH_FLOW_YIELD.value:
            free_cash_flow = utils.rolling_sum(
                fin_data[FundamentalDataTypes.freeCashFlow.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return free_cash_flow / (self.market_cap + eps)
        elif ratio_name == FundamentalRatios.CASH_FLOW_MARGIN.value:
            free_cash_flow = utils.rolling_sum(
                fin_data[FundamentalDataTypes.freeCashFlow.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)        
            total_revenue = utils.rolling_sum(
                fin_data[FundamentalDataTypes.totalRevenue.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return free_cash_flow / (total_revenue + eps)
        elif ratio_name == FundamentalRatios.CASH_FLOW_TO_NET_INCOME.value:
            free_cash_flow = utils.rolling_sum(
                fin_data[FundamentalDataTypes.freeCashFlow.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)        
            net_income = utils.rolling_sum(
                fin_data[FundamentalDataTypes.netIncome.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return free_cash_flow / (net_income + eps)
        elif ratio_name == FundamentalRatios.NET_PAYOUT_YIELD.value:
            net_payout = utils.rolling_sum(
                self._calc_net_payout(fin_data), 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return net_payout / (self.market_cap + eps)
        elif ratio_name == FundamentalRatios.EQUITY_ISSUANCE.value:
            shares = fin_data[FundamentalDataTypes.commonStockSharesOutstanding.value]
            return utils.rolling_return(shares, n_periods=n_periods, fillna=fillna,
                                        return_type=ReturnTypes.ARITHMETIC)
        elif ratio_name == FundamentalRatios.ASSETS_GROWTH.value:
            assets = fin_data[FundamentalDataTypes.totalAssets.value]
            return utils.rolling_return(assets, n_periods=n_periods, fillna=fillna, 
                                        return_type=ReturnTypes.ARITHMETIC)
        elif ratio_name == FundamentalRatios.BOOK_VALUE_GROWTH.value:
            book_value = fin_data[FundamentalDataTypes.totalStockholderEquity.value]
            return utils.rolling_return(book_value, n_periods=n_periods, fillna=fillna, 
                                        return_type=ReturnTypes.ARITHMETIC)
        elif ratio_name == FundamentalRatios.CAPEX_GROWTH.value:
            capex = fin_data[FundamentalDataTypes.capitalExpenditures.value]
            return utils.rolling_return(capex, n_periods=n_periods, fillna=fillna, 
                                        return_type=ReturnTypes.ARITHMETIC)
        elif ratio_name == FundamentalRatios.EARNINGS_GROWTH.value:
            earnings = utils.rolling_sum(
                fin_data[FundamentalDataTypes.netIncome.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return utils.rolling_return(earnings, n_periods=n_periods, fillna=fillna, 
                                        return_type=ReturnTypes.ARITHMETIC)
        elif ratio_name == FundamentalRatios.FIXED_ASSETS_GROWTH.value:
            ppe = fin_data[FundamentalDataTypes.propertyPlantEquipment.value]
            return utils.rolling_return(ppe, n_periods=n_periods, fillna=fillna, 
                                        return_type=ReturnTypes.ARITHMETIC)
        elif ratio_name == FundamentalRatios.NET_DEBT_GROWTH.value:
            net_debt = fin_data[FundamentalDataTypes.netDebt.value]
            return utils.rolling_return(net_debt, n_periods=n_periods, fillna=fillna,
                                        return_type=ReturnTypes.ARITHMETIC)
        elif ratio_name == FundamentalRatios.EARNINGS_TO_PRICE.value:
            earnings = utils.rolling_sum(
                fin_data[FundamentalDataTypes.netIncome.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return earnings / (self.market_cap + eps)
        elif ratio_name == FundamentalRatios.SALES_TO_EV.value:
            sales = utils.rolling_sum(
                fin_data[FundamentalDataTypes.totalRevenue.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            ev = self._calc_enterprise_value(fin_data)                
            return sales / (ev + eps)
        elif ratio_name == FundamentalRatios.SALES_TO_PRICE.value:
            sales = utils.rolling_sum(
                fin_data[FundamentalDataTypes.totalRevenue.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            return sales / (self.market_cap + eps)
        elif ratio_name == FundamentalRatios.BOOK_TO_PRICE.value:
            book_value = fin_data[FundamentalDataTypes.totalStockholderEquity.value]
            return book_value / (self.market_cap + eps)
        elif ratio_name == FundamentalRatios.EBITDA_TO_EV.value:
            ebitda = utils.rolling_sum(
                fin_data[FundamentalDataTypes.ebitda.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            ev = self._calc_enterprise_value(fin_data)
            return ebitda / (ev + eps)
        elif ratio_name == FundamentalRatios.FREE_CASH_FLOW_TO_EV.value:
            fcf = utils.rolling_sum(
                fin_data[FundamentalDataTypes.freeCashFlow.value],
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            ev = self._calc_enterprise_value(fin_data)
            return fcf / (ev + eps)
        else:
            raise ValueError(f'Unsupported fundamental ratio: {ratio_name}')

    def _calc_enterprise_value(self, fin_data):
        return self.market_cap \
            + fin_data[FundamentalDataTypes.longTermDebt.value] \
            + fin_data[FundamentalDataTypes.shortTermDebt.value] \
            - fin_data[FundamentalDataTypes.cashAndEquivalents.value]

    def _calc_net_payout(self, fin_data):
        divs = fin_data[FundamentalDataTypes.netIncome.value]
        sale_or_purchase_of_stock = fin_data[FundamentalDataTypes.salePurchaseOfStock.value]
        return divs - sale_or_purchase_of_stock

    def get_cta_momentum_signals(self):
        """Calculate the intermediate CTA Momentum signals following Baz et al for all symbols.

        This function calculates the 16 intermediate momentum signals, following
        the paper 'Dissecting Investment Strategies in the Cross Section and Time Series'
        by Baz et al, published in 2015.
        https://www.cmegroup.com/education/files/dissecting-investment-strategies-in-the-cross-section-and-time-series.pdf
        """
        if self._cta_momentum_signals is not None:
            return self._cta_momentum_signals

        self._cta_momentum_signals = dict()
        S = [8, 16, 32]
        L = [24, 48, 96]
        HL = lambda n : np.log(0.5) / np.log(1 - 1/n)

        signals = []
        signal_names = []
        for j in range(3):
            x = self.daily_prices.ewm(halflife=HL(S[j])).mean() - self.daily_prices.ewm(halflife=HL(L[j])).mean()
            y = x / np.maximum(self.daily_prices.rolling(63, min_periods=60).std(), self.price_tol)
            z = y / np.maximum(y.rolling(252, min_periods=240).std(), self.price_tol)
            u = x * np.exp(-np.power(x, 2) / 4) / 0.89    
            self._cta_momentum_signals['x' + str(j)] = x.resample('M').last().values
            self._cta_momentum_signals['y' + str(j)] = y.resample('M').last().values
            self._cta_momentum_signals['z' + str(j)] = z.resample('M').last().values
            self._cta_momentum_signals['u' + str(j)] = u.resample('M').last().values
        return self._cta_momentum_signals
