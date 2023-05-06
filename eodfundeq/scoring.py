import datetime
import numpy as np
import pandas as pd

from abc import abstractmethod, ABC
from statsmodels.stats.correlation_tools import cov_nearest 
from pypfopt import HRPOpt
from typing import Optional

from pyfintools.tools import optim
from secdb.strategy.core import RebalanceByWeights

from eodfundeq.constants import ModelTypes, ReturnTypes
from eodfundeq import utils


def get_selected_log_returns(model, dataset, daily_prices, n_stocks=100, price_tol=1e-6):
    return get_selected_returns(
        model, dataset, daily_prices, return_type=ReturnTypes.LOG, n_stocks=n_stocks, price_tol=price_tol)

def get_selected_arith_returns(model, dataset, daily_prices, n_stocks=100, price_tol=1e-6):
    return get_selected_returns(
        model, dataset, daily_prices, return_type=ReturnTypes.ARITHMETIC, n_stocks=n_stocks, price_tol=price_tol)

def get_selected_returns(model, dataset, daily_prices, return_type, n_stocks=100, price_tol=1e-6):
    assert return_type in ReturnTypes
    timestamps = dataset.timestamp
    true_rtns_list = []
    selected_symbols_list = []
    period_ends = []
    sorted_timestamps = sorted(set(timestamps))
    for timestamp in sorted_timestamps:
        idx_t = (timestamp == timestamps)
        symbols_t = dataset.symbol.loc[idx_t]        
        y_score = model.predict(dataset.X.loc[idx_t,:])
        df_scores = pd.Series(y_score.flatten(), index=symbols_t)
        df_scores.sort_values(inplace=True, ascending=False)
        selected_symbols = df_scores.index.values[:n_stocks]
        period_end = timestamp + pd.tseries.offsets.MonthEnd(0)
        period_ends.append(period_end)
        period_start = period_end - pd.Timedelta(91, unit='d')
        idx_period = (period_start <= daily_prices.index) & (daily_prices.index <= period_end)
        period_prices = daily_prices.loc[idx_period, selected_symbols]
        price_ratio = np.maximum(period_prices, price_tol) / \
                      np.maximum(period_prices.shift(1), price_tol)
        if return_type == ReturnTypes.LOG:
            rtns = np.log(price_ratio)
        else:
            rtns = -1 + price_ratio
        rtns = rtns.iloc[1:,:]
        assert np.isnan(rtns.values).sum(axis=0).max() < 10
        true_rtns = pd.Series(dataset.true_return.loc[idx_t].values, index=symbols_t)
        true_rtns_list.append(true_rtns.loc[selected_symbols].values)
        selected_symbols_list.append(selected_symbols)

    df_rtns = pd.DataFrame(true_rtns_list, index=pd.DatetimeIndex(period_ends))
    df_symbols = pd.DataFrame(selected_symbols_list, index=pd.DatetimeIndex(period_ends))
    return df_rtns, df_symbols

def get_strategy_weights(model, dataset, daily_prices, target_vol=0.15, n_stocks=100, weighting='equal'):
    _, df_symbols = get_selected_arith_returns(model, dataset, daily_prices, n_stocks=n_stocks)
    weights_list = []
    for period_end in df_symbols.index:
        if weighting == 'equal':
            weights = np.ones((n_stocks,)) / n_stocks
        else:
            period_symbols = df_symbols.loc[period_end].values
            asset_cov = _calculate_covariance(daily_prices, period_end, period_symbols, n_days=91)
            if weighting == 'minvar':
                weights, _ = optim.optimize_min_variance(asset_cov, ub=0.05)
                assert np.isclose(weights.sum(), 1.0, atol=0.01), 'Weights must sum to 1.0'
            elif weighting == 'vol':
                realized_vol = np.diag(asset_cov)
                w_0 = 1 / (n_stocks * np.minimum(realized_vol, 0.03))
                vol = np.sqrt(w_0 @ asset_cov @ w_0)
                weights = w_0 * target_vol / vol
                if weights.sum() > 1:
                    weights /= weights.sum()
            elif weighting == 'erc':
                lower_bound = 1/n_stocks * 0.5
                upper_bound = 1/n_stocks * 2.0
                weights = optim.optimize_erc(
                    asset_cov, vol_lb=target_vol, vol_ub=target_vol,
                    lb=lower_bound, ub=upper_bound, unit_constraint=True)
            else:
                raise ValueError(f'Unsupported weighting scheme: {weighting}')
        weights_list.append(weights)
    return pd.DataFrame(np.vstack(weights_list), index=df_symbols.index)

def _calculate_covariance(daily_prices, period_end, symbols, n_days=91, price_tol=1e-6):
    period_start = period_end - pd.Timedelta(n_days, unit='d')
    idx_period = (period_start <= daily_prices.index) & (daily_prices.index <= period_end)
    period_prices = daily_prices.loc[idx_period, symbols]
    rtns = np.log(np.maximum(period_prices, price_tol) / \
                  np.maximum(period_prices.shift(1), price_tol))
    rtns = rtns.iloc[1:,:]
    assert np.isnan(rtns.values).sum(axis=0).max() < 10

    # Calculate covariance with pandas, which may be non positive definite if there are NaNs
    asset_cov_non_pos_def = rtns.cov() * 252

    # Shift to nearest positive definite covariance
    asset_cov = cov_nearest(asset_cov_non_pos_def, threshold=1e-6)
    return asset_cov

def get_performance(model, dataset, daily_prices, return_window,
                    n_stocks=100, weighting='equal'):
    _, df_symbols = get_selected_log_returns(
        model, dataset, daily_prices, n_stocks=n_stocks)
    target_wts = get_strategy_weights(model, dataset, daily_prices, weighting=weighting)
    df_weights = get_portfolio_weights(target_wts, df_symbols, return_window)

    # We exclude stocks from selection if they have low or NaN prices.
    # However, a stock that is selected could have its price become NaN
    # during the investment period. In this case, we just use the last
    # available price until the end of the period.
    df_prices = daily_prices[df_weights.columns.values].ffill()

    # Align dates of weights with pricing dates
    idx_dt = df_prices.index.get_indexer(df_weights.index, 'ffill')
    df_weights.index = df_prices.index[idx_dt]

    ptf = RebalanceByWeights(weights=df_weights, prices=df_prices)
    ptf.initial_value = 1.0
    mv_ts, _ = ptf.calc_performance(
        start=df_weights.index[0],
        end=df_weights.index[-1] + pd.tseries.offsets.MonthEnd(return_window))
    return mv_ts

def get_portfolio_weights(target_wts, df_symbols, return_window):
    uniq_symbols = sorted(set(df_symbols.values.flatten()))
    dates = list(df_symbols.index.values)
    for j in range(1, return_window):
        dates.append(dates[-1] + pd.tseries.offsets.MonthEnd(1))

    wts_init_arr = np.zeros((df_symbols.shape[0] + return_window - 1,
                             len(uniq_symbols)), dtype=np.float32)
    df_weights = pd.DataFrame(wts_init_arr, index=dates, columns=uniq_symbols)
    for idx in target_wts.index:
        for j in range(return_window):
            idx_t = idx + pd.tseries.offsets.MonthEnd(j)
            df_weights.loc[idx_t, df_symbols.loc[idx]] += target_wts.loc[idx].values

    return df_weights / df_weights.sum(axis=1).values.reshape(-1, 1)

def get_ndcg(model, dataset, n_buckets, n_stocks=100):
    timestamps = dataset.timestamp
    model_period_ndcg = []
    sorted_timestamps = sorted(set(timestamps))
    for timestamp in sorted_timestamps:
        idx_t = timestamp == timestamps
        y_score = model.predict(dataset.X.loc[idx_t,:]).flatten()
        y_true = dataset.y_cls.loc[idx_t].values
        if model.direction == ModelTypes.BEAR:
            y_true = n_buckets - 1 - y_true
        ndcg_val = utils.calc_ndcg(y_true, y_score, k=n_stocks, form='exp')
        assert ndcg_val > 0
        model_period_ndcg.append(ndcg_val)
    return np.array(model_period_ndcg)