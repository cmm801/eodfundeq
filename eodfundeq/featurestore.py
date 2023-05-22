"""Define methods for analyzing fundamental data from eodhistoricaldata.com.

This file contains the core logic used for the analysis.
"""

import numpy as np
import pandas as pd
import pandas_market_calendars
import scipy.stats

from collections.abc import Iterable
from enum import Enum

import findatadownload
import pyfintools.tools.freq
from pyfintools.tools import tradingutils

from eodfundeq import utils
from eodfundeq.constants import ReturnTypes, FundamentalRatios, FUNDAMENTAL_RATIO_INPUTS, RFR_SYMBOL, TSNames, TRADING_DAYS_PER_MONTH, TRADING_DAYS_PER_YEAR

from eodhistdata import EODHelper, FundamentalEquitySnapshot, FundamentalEquityTS
from eodhistdata.constants import FundamentalDataTypes, TimeSeriesNames


TIME_SERIES_PANEL_INFO = {
    TSNames.ADJUSTED_CLOSE.value: dict(dtype=np.float32, frequency='m'),
    TSNames.CLOSE.value: dict(dtype=np.float32, frequency='m'),
    TSNames.DAILY_PRICES.value: dict(dtype=np.float32, frequency='b'),
    TSNames.DAILY_VOLUME.value: dict(dtype=np.float32, frequency='b'),
    TSNames.MARKET_CAP.value: dict(dtype=np.float32, frequency='m'),
    TSNames.VOLUME.value: dict(dtype=np.float32, frequency='m'),
}

class FeatureStore(object):
    def __init__(self, api_token, base_path, start, end='', symbols=None, 
                 n_val_periods=24, n_test_periods=24, stale_days=240, drop_empty_ts=True):
        self.eod_helper = EODHelper(api_token=api_token, base_path=base_path)
        self.n_val_periods = n_val_periods        
        self.n_test_periods = n_test_periods
        self.stale_days = stale_days
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

        # Initialize container for caching calculated volatility
        self._volatility = dict()

        # Initialize container for caching calculated fundamental ratios
        self._fundamental_ratios = None        

        # Other parameters
        self.price_tol = 1e-4           # Used to make sure we don't divide by 0
        self.fundamental_data_delay = 1 # Months to delay use of fundamental data, to account for
                                        # the fact that it is not immediately available for trading. 

    @property
    def fundamental_ratios(self):
        return self._fundamental_ratios

    def _init_data_rows(self, n_rows, dtype=np.float32):
        return np.ones((n_rows, self.symbols.size), dtype=dtype)

    def init_data_panel(self, dtype=np.float32, frequency='m'):
        return self._init_data_rows(self.dates[frequency].size, dtype=dtype)

    def init_pd_dataframe(self, frequency, dtype=np.float32):
        return pd.DataFrame(
            np.ones((self.dates[frequency].size, self.symbols.size), dtype=dtype),
            index=self.dates[frequency], columns=self.symbols)

    def _get_loc(self, date):
        try:
            return self._date_map[date]
        except KeyError:
            return self._date_map[pd.Timestamp(date) + pd.tseries.offsets.MonthEnd(0)]

    def load_ohlcv_data(self):
        ts_types = (TSNames.ADJUSTED_CLOSE, TSNames.VOLUME, TSNames.CLOSE,
                    TSNames.DAILY_PRICES, TSNames.DAILY_VOLUME)
        if np.all([t in self._time_series for t in ts_types]):
            return  # Time series are already loaded

        for name in ts_types:
            kwargs = TIME_SERIES_PANEL_INFO[name.value]
            default_val = 0.0 if name.value == TSNames.VOLUME.value else np.nan
            self._time_series[name.value] = default_val * self.init_pd_dataframe(**kwargs)

        for _, symbol in enumerate(self.symbols):
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
            self._time_series[TSNames.DAILY_VOLUME.value].loc[daily_ts.index[idx_shared], symbol] = \
                    daily_ts.volume.values[idx_shared].astype(np.float32)

            # Downsample to monthly data (from daily)
            monthly_ts = tradingutils.downsample(daily_ts, frequency='M')
            if monthly_ts.shape[0] < 2:
                continue

            # Add each time series' data to the respective panel
            for ts_enum in (TSNames.ADJUSTED_CLOSE, TSNames.CLOSE, TSNames.VOLUME):
                idx_shared = monthly_ts.index.isin(self.dates['m'])
                self._time_series[ts_enum.value].loc[monthly_ts.index[idx_shared], symbol] = \
                    monthly_ts[ts_enum.value].values[idx_shared].astype(np.float32)

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
        self._time_series[mc_name] = np.nan * self.init_pd_dataframe(**kwargs)
        for _, symbol in enumerate(self.symbols):
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
    def daily_volume(self):
        if not len(self._time_series):
            self.load_ohlcv_data()
        return self._time_series[TSNames.DAILY_VOLUME.value]

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

    def get_future_realized_volatility(self, forecast_horizon: int,
                                       min_fraction: float = 0.9, frequency: str = 'm'):
        realized_vol = pd.DataFrame(self.get_realized_volatility(
            window=forecast_horizon, min_fraction=min_fraction, frequency=frequency))
        return realized_vol.shift(-forecast_horizon).values

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

    def get_log_price_returns(self, frequency='b'):
        """Get the log price returns for a specified frequency."""
        if frequency == 'm':
            prices = self.adjusted_close
        else:
            prices = self.daily_prices
        log_returns = np.log(np.maximum(prices, self.price_tol) /  \
                             np.maximum(prices.shift(1).values, self.price_tol))
        log_returns[np.isclose(prices.values, 0)] = np.nan
        log_returns[np.isclose(prices.shift(1).values, 0)] = np.nan
        return log_returns

    def get_realized_volatility(self, window: int, min_fraction=0.9, frequency='m'):
        """Calculate the realized volatility over rolling windows."""
        # Get the daily prices, and convert to target frequency afterwards
        daily_log_rtns = self.get_log_price_returns(frequency='b')
        minp = int(window * min_fraction)
        rolling_std = daily_log_rtns.rolling(window, axis=0, min_periods=minp).std()
        rolling_vol = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if frequency == 'm':
            rolling_vol = rolling_vol.resample('M').last()
        if ~np.all(rolling_vol.index.values == self.dates[frequency]):
            raise ValueError('Unexpected dates found in rolling vol.')
        return rolling_vol.values

    def get_risk_free_rate(self, maturity='1m', frequency='b'):
        """Get a time series for the risk-free rate."""
        if maturity == '1m':
            base_ticker='DGS1MO'
        else:
            raise NotImplementedError('Only implemented for maturity of 1-month')
        return self._get_fred_ts(base_ticker, frequency)

    def get_vix(self, frequency='b'):
        """Get time series for VIX."""
        base_ticker = 'VIXCLS'
        return self._get_fred_ts(base_ticker, frequency)

    def get_us_equity_tr_index(self, frequency='b'):
        """Get time series for US equity total return Index (Wilshire 5000)"""
        base_ticker = 'WILL5000IND'
        return self._get_fred_ts(base_ticker, frequency)

    def create_panel_from_time_series(self, ts_vals):
        """Create a data panel with copies of a time series."""
        if isinstance(ts_vals, (pd.Series, pd.DataFrame)):
            ts_vals = ts_vals.values
        return np.tile(ts_vals.reshape(-1, 1), self.symbols.size)

    def create_panel_from_row(self, row_data, frequency):
        """Create a data panel with copies of a data row at a point in time."""
        return np.tile(row_data.reshape(-1, 1), len(self.dates[frequency])).T

    def _get_fred_ts(self, base_ticker, frequency):
        ts = findatadownload.download_time_series(
            data_source='fred', base_ticker=base_ticker, start=self.start)
        ts = ts.reindex(self.dates[frequency])
        ts = ts.squeeze()  # Convert to pandas Series
        ts.name = base_ticker
        return ts

    def get_risk_free_returns(self, maturity='1m'):
        """Get a time series representing returns from investing in risk-free rate"""
        rfr_ts = self.get_risk_free_rate(maturity=maturity)
        rfr_ts.name = RFR_SYMBOL

        # Need to accumulate extra returns based on elapsed days
        # (e.g., we earn risk-free-rate for 3 days over weekend but 1 day otherwise)
        n_day_diff = pd.Series(np.hstack([
            [np.nan], 
            np.array((rfr_ts.index[1:] - rfr_ts.index[:-1]) / pd.Timedelta('1d'))
        ]), index=rfr_ts.index)
        return -1 + np.power(1 + rfr_ts.shift(1) / 365 / 100, n_day_diff)

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
            fin_data[fin_data_type] = np.nan * self.init_data_panel()

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

    def calc_all_fundamental_ratios(self, fillna=False, n_periods=None, min_obs=4):
        if self._fundamental_ratios is None:
            self._fundamental_ratios = dict()
            fin_data = self.get_financial_statement_input_data()            
            for ratio_type in FundamentalRatios:
                self._fundamental_ratios[ratio_type.value] = self.calculate_fundamental_ratio(
                    ratio_type.value, fin_data, n_periods=n_periods, min_obs=min_obs, fillna=fillna)
        return self._fundamental_ratios

    def calculate_fundamental_ratio(self, ratio_name, fin_data, n_periods=None,
                                     min_obs=4, fillna=False, frequency='m'):
        n_periods = int(pyfintools.tools.freq.get_periods_per_year(frequency))
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
