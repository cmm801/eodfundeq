import datetime
import numpy as np
import pandas as pd

from abc import abstractmethod, ABC
from statsmodels.stats.correlation_tools import cov_nearest 
from pypfopt import HRPOpt
from typing import Optional

from pyfintools.tools import optim

from eodfundeq.constants import START_DATE, DatasetTypes, ModelTypes
from eodfundeq import filters
from eodfundeq import utils


def get_selected_returns(model, dataset, daily_prices, n_stocks=100, price_tol=1e-6):
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
        rtns = np.log(np.maximum(period_prices, price_tol) / \
                    np.maximum(period_prices.shift(1), price_tol))
        rtns = rtns.iloc[1:,:]
        assert np.isnan(rtns.values).sum(axis=0).max() < 10
        true_rtns = pd.Series(dataset.true_return.loc[idx_t].values, index=symbols_t)
        true_rtns_list.append(true_rtns.loc[selected_symbols].values)
        selected_symbols_list.append(selected_symbols)

    df_rtns = pd.DataFrame(true_rtns_list, index=pd.DatetimeIndex(period_ends))
    df_symbols = pd.DataFrame(selected_symbols_list, index=pd.DatetimeIndex(period_ends))
    return df_rtns, df_symbols

def get_strategy_weights(model, dataset, daily_prices, target_vol=0.15, n_stocks=100, weighting='equal'):
    _, df_symbols = get_selected_returns(model, dataset, daily_prices, n_stocks=n_stocks)
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
                weights = 1/n_stocks * np.minimum(target_vol / realized_vol, 1.2)
            elif weighting == 'erc':
                lb = 1/n_stocks * 0.5
                ub = 1/n_stocks * 2.0
                weights = optim.optimize_erc(asset_cov, vol_lb=target_vol, vol_ub=target_vol,
                    lb=lb, ub=ub, unit_constraint=True)
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

def get_performance(model, dataset, daily_prices, n_stocks=100, weighting='equal'):
    df_rtns, _ = get_selected_returns(model, dataset, daily_prices, n_stocks=n_stocks)
    df_wts = get_strategy_weights(model, dataset, daily_prices, n_stocks=n_stocks, weighting=weighting)
    realized_rtns = []
    for period_end in df_rtns.index:
        weights = df_wts.loc[period_end].values
        true_rtns = df_rtns.loc[period_end].values
        realized_rtns.append((weights * true_rtns).sum())
    return pd.Series(realized_rtns, index=df_rtns.index)

def calc_pro_rata_performance(rtns, return_window):
    dti = pd.DatetimeIndex(rtns.index)
    start = rtns.index[0] - pd.tseries.offsets.MonthEnd(1)
    strat_perf = pd.DataFrame(np.nan * np.ones((rtns.shape[0]+1, return_window), dtype=float), 
                            index=dti.append(pd.DatetimeIndex([start])).sort_values())
    strat_perf.iloc[0] = 1/return_window
    for j in range(rtns.shape[0]):
        for col in range(return_window):
            if col == j % return_window:
                strat_perf.iloc[j+1, col] = strat_perf.iloc[j, col] * (1 + rtns.iloc[j])

            else:
                strat_perf.iloc[j+1, col] = strat_perf.iloc[j, col]
    total_perf = strat_perf.sum(axis=1)
    total_perf.name = rtns.name
    return total_perf

def get_ndcg(model, dataset, n_buckets, n_stocks=100):
    timestamps = dataset.timestamp
    model_period_ndcg = []
    sorted_timestamps = sorted(set(timestamps))
    for timestamp in sorted_timestamps:
        idx_t = (timestamp == timestamps)
        y_score = model.predict(dataset.X.loc[idx_t,:]).flatten()
        y_true = dataset.y_cls.loc[idx_t].values
        if model.direction == ModelTypes.BEAR:
            y_true = n_buckets - 1 - y_true
        ndcg_val = utils.calc_ndcg(y_true, y_score, k=n_stocks, form='exp')
        assert ndcg_val > 0
        model_period_ndcg.append(ndcg_val)
    return np.array(model_period_ndcg) 