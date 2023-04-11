"""Define methods for analyzing fundamental data from eodhistoricaldata.com.

This file contains the core logic used for the analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats

from collections.abc import Iterable
from enum import Enum
from typing import Optional, Union

import pyfintools.tools.freq
from pyfintools.tools import tradingutils

from eodfundeq import utils
from eodfundeq.constants import ReturnTypes, DataSetTypes, FundamentalRatios, FUNDAMENTAL_RATIO_INPUTS
from eodfundeq.filters import EqualFilter, InRangeFilter, EntireColumnInRangeFilter, IsNotNAFilter
from eodfundeq.timeseriespanel import TimeSeriesPanel

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
    TSNames.ADJUSTED_CLOSE.value: dict(default_val=np.nan, dtype=np.float32, frequency='m'),
    TSNames.CLOSE.value: dict(default_val=np.nan, dtype=np.float32, frequency='m'),
    TSNames.DAILY_PRICES.value: dict(default_val=np.nan, dtype=np.float32, frequency='b'),
    TSNames.MARKET_CAP.value: dict(default_val=np.nan, dtype=np.float32, frequency='m'),    
    TSNames.MONTHLY_RETURNS.value: dict(default_val=np.nan, dtype=np.float32, frequency='m'),
    TSNames.VOLUME.value: dict(default_val=0.0, dtype=np.float32, frequency='m'),
}

class StockFeatureAnalyzer(object):
    periods_per_year = 12

    def __init__(self, api_token, base_path, start, end='', symbols=None, 
                 n_val_periods=24, n_test_periods=24, stale_days=120, clip=None):
        self.eod_helper = EODHelper(api_token=api_token, base_path=base_path)
        self.n_val_periods = n_val_periods        
        self.n_test_periods = n_test_periods
        self.stale_days = stale_days
        self.clip = clip if clip is not None else (-np.inf, np.inf)

        self.start = pd.Timestamp(start)
        if not end:
            self.end = pd.Timestamp.now().round('d') - pd.tseries.offsets.MonthEnd(1)
        else:
            self.end = pd.Timestamp(end)

        self.start_str = self.start.strftime('%Y-%m-%d')
        self.end_str = self.end.strftime('%Y-%m-%d')

        self.dates = pd.date_range(self.start, self.end, freq='M')
        if symbols is None:
            self.symbols = self.eod_helper.get_non_excluded_exchange_symbols('US')
        else:
            self.symbols = symbols

        # Initialize the time series panels
        self._time_series = dict()

        # Initialize cached time series data
        self._cta_momentum_signals = None
        
        # initialize hash table to look up date locations
        self._date_map_pd = pd.Series([], dtype=int)
        self._date_map_str = pd.Series([], dtype=int)
        for j, dt in enumerate(self.dates):
            self._date_map_pd[dt] = j
            self._date_map_str[dt.strftime('%Y-%m-%d')] = j
            
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
        self.filter_min_monthly_volume = 21 * 5000  # Exclude stocks with low trading volume
        self.filter_max_return = 100  # Excludes return outliers from sample

        # Set default filters
        self.set_default_filters()

    def _create_data_rows(self, n_rows, dtype=np.float32):
        return np.ones((n_rows, self.n_symbols), dtype=dtype)

    def _create_data_panel(self, dtype=np.float32):
        return self._create_data_rows(self.n_dates, dtype=dtype)

    def _create_ts_panel(self, frequency, dtype=np.float32, default_val=None):
        return TimeSeriesPanel(self.symbols, start=self.start, end=self.end,
            frequency=frequency, dtype=dtype, default_val=default_val)

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

    def load_ohlcv_data(self):
        ts_types = (TSNames.ADJUSTED_CLOSE, TSNames.CLOSE, TSNames.DAILY_PRICES,
                    TSNames.MONTHLY_RETURNS, TSNames.VOLUME)
        if np.all([t in self._time_series for t in ts_types]):
            return  # Time series are already loaded

        for name in ts_types:
            kwargs = TIME_SERIES_PANEL_INFO[name.value]
            self._time_series[name.value] = self._create_ts_panel(**kwargs)

        for idx_symbol, symbol in enumerate(self.symbols):
            daily_ts = self.eod_helper.get_historical_data(
                symbol, start=self.start, stale_days=self.stale_days)
            if not daily_ts.shape[0]:
                continue

            # Add daily prices (adjusted closing price) to the panel
            self._time_series[TSNames.DAILY_PRICES.value].add_time_series(
                daily_ts[TSNames.ADJUSTED_CLOSE.value], idx_symbol)

            # Downsample to monthly data (from daily)
            monthly_ts = tradingutils.downsample(daily_ts, frequency='M')
            monthly_ts = monthly_ts.loc[(self.start <= monthly_ts.index) &
                                        (monthly_ts.index <= self.end)]
            if monthly_ts.shape[0] < 2:
                continue

            # Add each time series' data to the respective panel
            for ts_enum in (TSNames.ADJUSTED_CLOSE, TSNames.CLOSE, TSNames.VOLUME):
                self._time_series[ts_enum.value].add_time_series(
                    monthly_ts[ts_enum.value], idx_symbol)
        
        # Fill non-trailing NaNs in close and adjusted_close time series
        self._time_series[TSNames.CLOSE.value].ffill()
        self._time_series[TSNames.ADJUSTED_CLOSE.value].ffill()
        
        # Also add monthly returns
        adj_close_panel = self._time_series[TSNames.ADJUSTED_CLOSE.value].data
        self._time_series[TSNames.MONTHLY_RETURNS.value].set_data(np.vstack([
            self._create_data_rows(1),
            -1 + adj_close_panel[1:,:] / (adj_close_panel[:-1,:] + self.price_tol)
        ]))

    def load_market_cap_data(self):
        mc_name = TSNames.MARKET_CAP.value
        kwargs = TIME_SERIES_PANEL_INFO[mc_name]
        if mc_name in self._time_series:
            return  # Time series already loaded
        self._time_series[mc_name] = self._create_ts_panel(**kwargs)
        for idx_symbol, symbol in enumerate(self.symbols):
            raw_ts = self.eod_helper.get_market_cap(
                symbol, start=self.start, stale_days=self.stale_days)
            if not raw_ts.shape[0]:
                continue
            monthly_ts = raw_ts.resample('M').last()
            monthly_ts = monthly_ts.loc[(self.start <= monthly_ts.index) &
                                        (monthly_ts.index <= self.end)]
            if monthly_ts.shape[0] > 2:
                self._time_series[mc_name].add_time_series(monthly_ts, idx_symbol)

    def set_default_filters(self):
        self._good_mask = None
        self._filters = [
            IsNotNAFilter(self, TSNames.CLOSE.value),
            IsNotNAFilter(self, TSNames.ADJUSTED_CLOSE.value),
            IsNotNAFilter(self, TSNames.VOLUME.value),
            InRangeFilter(self, TSNames.CLOSE.value,
                          low=self.filter_min_price),
            InRangeFilter(self, TSNames.ADJUSTED_CLOSE.value,
                          low=self.filter_min_price),
            InRangeFilter(self, TSNames.VOLUME.value,
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
    def n_dates(self):
        return self.dates.size
    
    @property
    def n_symbols(self):
        return len(self.symbols)
        
    @property
    def filters(self):
        return self._filters

    @property
    def good_mask(self):
        if self._good_mask is None:
            self._good_mask = self._create_data_panel(dtype=bool)
            for f in self.filters:
                self._good_mask &= f.get_mask()

        return self._good_mask

    @property
    def adjusted_close(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.ADJUSTED_CLOSE.value].data
        
    @property
    def close(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.CLOSE.value].data

    @property
    def daily_prices(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.DAILY_PRICES.value].data

    @property
    def monthly_returns(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.MONTHLY_RETURNS.value].data

    @property
    def volume(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.VOLUME.value].data

    @property
    def market_cap(self):
        if TSNames.MARKET_CAP.value not in self._time_series:
            self.load_market_cap_data()
        return self._time_series[TSNames.MARKET_CAP.value].data
    
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

    def get_volatility(self, windows: Union[int, Iterable], return_type=ReturnTypes.LOG.value):
        if not isinstance(windows, Iterable):
            windows = [windows]

        missing_windows = []
        for window in windows:
            vol_name = f'volatility_{window}m'
            if vol_name not in self._volatility:
                self._volatility[vol_name] = np.nan * self._create_data_panel()
                missing_windows.append(window)

        if not missing_windows:
            return self._volatility

        for idx_symbol, symbol in enumerate(self.symbols):
            daily_ts = self.eod_helper.get_historical_data(
                symbol, start=self.start, stale_days=self.stale_days)
            daily_ts = daily_ts.loc[(self.start <= daily_ts.index) &
                                    (daily_ts.index <= self.end)]
            if daily_ts.shape[0] < 2:
                continue
            
            daily_price_ts = daily_ts[TSNames.ADJUSTED_CLOSE.value]
            if return_type == ReturnTypes.LOG.value:
                daily_return_ts = np.log(daily_price_ts / (daily_price_ts.shift(1).values + self.price_tol))
            elif return_type == ReturnTypes.ARITHMETIC.value:
                daily_return_ts = -1 + daily_price_ts / (daily_price_ts.shift(1).values + self.price_tol)
            else:
                raise ValueError(f'Unsupported return type: {return_type}')

            for vol_window in missing_windows:
                vol_name = f'volatility_{vol_window}m'
                w = vol_window * TRADING_DAYS_PER_MONTH
                daily_vol_ts = daily_return_ts.rolling(w).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                monthly_vol_ts = daily_vol_ts.resample('M').last()

                # Add each time series' data to the respective panel
                idx_date = self._get_loc(monthly_vol_ts.index[0])
                if idx_date < 0:
                    raise ValueError(f'Missing date: {monthly_vol_ts.index[0]}')
                L = monthly_vol_ts.shape[0]
                self._volatility[vol_name][idx_date:idx_date+L, idx_symbol] = monthly_vol_ts.values
        return self._volatility

    def get_financial_statement_input_data(self):
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
            fin_data[fin_data_type] = np.nan * self._create_data_panel()

        # For each symbol in our universe, get time series for all required input data types
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
                    idx_series = self._date_map_str[ts.index]
                except KeyError:
                    # If one of the dates is not in the target index, try moving dates to month-end
                    dates_pd = pd.DatetimeIndex(ts.index) + pd.tseries.offsets.MonthEnd(0)
                    idx_series = self._date_map_pd[dates_pd]

                idx = self.fundamental_data_delay + idx_series.values
                fin_data[fin_data_type][idx[idx < self.n_dates], idx_symbol] = ts.values[idx < self.n_dates]
        return fin_data

    def bucket_results(self, metric_vals, return_window, clip=None):
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
        for idx_date, date in enumerate(self.dates):
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

    def get_buckets(self, metric_vals):
        buckets = np.nan * np.ones_like(metric_vals, dtype=np.int32)
        for idx_date, date in enumerate(self.dates):
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
            begin_period_assets = np.vstack([np.nan * self._create_data_rows(n_periods), 
                                            total_assets[:-n_periods,:]])
            return net_income / (begin_period_assets + eps)
        elif ratio_name == FundamentalRatios.ROE.value:
            net_income = utils.rolling_sum(
                fin_data[FundamentalDataTypes.netIncome.value], 
                n_periods=n_periods, min_obs=min_obs, fillna=fillna)
            equity = fin_data[FundamentalDataTypes.totalStockholderEquity.value]
            begin_period_equity = np.vstack([np.nan * self._create_data_rows(n_periods),
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
            begin_period_capital = np.vstack([np.nan * self._create_data_rows(n_periods),
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
        for idx_symbol, symbol in enumerate(self.symbols):
            daily_ts = self.eod_helper.get_historical_data(
                symbol, start=self.start, stale_days=self.stale_days)
            
            daily_price_ts = daily_ts[TSNames.ADJUSTED_CLOSE.value]
            df_signals = utils.calc_cta_momentum_signals(daily_price_ts)
            df_signals = df_signals.loc[(self.start <= df_signals.index) &
                                    (df_signals.index <= self.end)]

            if df_signals.shape[0] < 2:
                continue

            # Add each time series' data to the respective panel
            idx_date = self._get_loc(df_signals.index[0])
            assert idx_date >= 0, 'Missing date'
            L = df_signals.shape[0]

            # Initialize a data panel for each sub-signal if necessary
            if not len(self._cta_momentum_signals):
                for col in df_signals.columns:
                    self._cta_momentum_signals[col] = np.nan * self._create_data_panel()

            for col in df_signals.columns:
                self._cta_momentum_signals[col][idx_date:idx_date+L, idx_symbol] = df_signals[col].values
        return self._cta_momentum_signals
