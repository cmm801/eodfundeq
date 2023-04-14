import datetime
import numpy as np
import pandas as pd
import warnings

from pypfopt import HRPOpt

from eodfundeq.constants import ReturnTypes
from pyfintools.tools import optim


def ffill(arr):
    """Method to front-fill NaNs in a numpy array."""
    mask = np.isnan(arr)

    # Get a matrix with index of valid data point, and 0 otherwise
    idx = np.where(~mask, np.arange(mask.shape[0]).reshape(-1, 1), 0)

    # Adjuste index so it contains the row index of the last non-NaN point
    np.maximum.accumulate(idx, axis=0, out=idx)

    # Add additional mask so NaNs beyond last valid data point don't get filled
    row_max = np.max(idx, axis=0)
    row_num = np.arange(idx.shape[0]).reshape(-1, 1)
    traing_nan_mask = (idx == row_max) & (idx < row_num)
    mask = mask & ~traing_nan_mask

    # Replace non-trailing NaNs with previous value
    arr[mask] = arr[idx[mask], np.nonzero(mask)[1]]
    return arr    

def rolling_sum(arr, n_periods, min_obs=None, fillna=True):
    """Compute a rolling sum of a numpy array."""
    if min_obs is None:
        min_obs = n_periods

    cs = np.nancumsum(arr, axis=0)
    roll_sum = np.vstack([
            cs[:n_periods, :],
            cs[n_periods:, :] - cs[:arr.shape[0]-n_periods, :]
    ]).astype(np.float32)
    
    observations = ~np.isnan(arr)
    cs_obs_window = np.cumsum(observations, axis=0)
    obs_in_window = np.vstack([
            cs_obs_window[:n_periods, :],
            cs_obs_window[n_periods:, :] - cs_obs_window[:arr.shape[0]-n_periods, :]
    ])
    roll_sum[obs_in_window < min_obs] = np.nan
    
    # Re-fill values that were originally NaNs
    if not fillna:
        roll_sum[np.isnan(arr)] = np.nan
    return roll_sum

def rolling_mean(arr, n_periods, min_obs=None, fillna=True):
    rsum = rolling_sum(arr, n_periods, min_obs=min_obs, fillna=fillna)
    n_obs = rolling_sum(~np.isnan(arr), n_periods, min_obs=1, fillna=fillna)
    return rsum / n_obs

def rolling_std(arr, n_periods, min_obs=None, fillna=True):
    """Calculate the rolling standard deviation.

    This should match the numpy calculation:
        std = np.sqrt(np.mean(x - np.mean(x)))
    """
    mu = rolling_mean(arr, n_periods, min_obs=min_obs)
    return np.sqrt(rolling_mean(arr ** 2, n_periods, min_obs=min_obs) - \
                   2 * mu * rolling_mean(arr, n_periods, min_obs=min_obs) + mu ** 2)

def rolling_return(arr, n_periods, return_type=None, fillna=True, tol=1e-6):
    """Compute a rolling return of a numpy array."""
    if return_type is None:
        return_type = ReturnTypes.ARITHMETIC

    if fillna:
        arr = ffill(arr)

    ratio =  arr[n_periods:,:] / (arr[:-n_periods,:] + tol)

    if return_type == ReturnTypes.ARITHMETIC:
        rtn_vals = -1 + ratio
    elif return_type == ReturnTypes.LOG:
        rtn_vals = np.log(ratio)        
    else:
        raise ValueError(f'Unknown return type: {return_type}.')

    empty_rows = np.nan * np.ones((n_periods, arr.shape[1]), dtype=arr.dtype)
    return np.vstack([empty_rows, rtn_vals])

def calc_feature_percentiles(array):
    """Calculate the percentiles across rows for an array of features."""
    order = array.argsort()
    ranks = order.argsort().astype(np.float32)
    ranks[np.isnan(array)] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        percentiles = ranks / np.nanmax(ranks, axis=1, keepdims=True)
    return percentiles

def ndcg(y_true, y_score, k=None, form='exp'):
    df = pd.DataFrame(np.vstack([y_score, y_true]).T, columns=['score', 'true'])
    df.sort_values('score', ascending=False, inplace=True)
    if k is None:
        k = df.shape[0]
    k = min(k, df.shape[0])
    relevance = df.true.values[:k]
    ideal_rel = df.sort_values('true', ascending=False).true.values[:k]
    discount = np.log2(np.arange(1, k+1) + 1)
    if form in ('exp', 'exponential'):
        DCG = (2 ** relevance - 1) / discount
        IDCG = (2 ** ideal_rel - 1) / discount
    else:
        DCG = relevance / discount
        IDCG = ideal_rel / discount

    return DCG.sum() / IDCG.sum()    

def cross_validate_gbm(model_handle, ds_helper, n_folds=5, **kwargs):
    X = ds_helper.X['train']
    y = ds_helper.y_class_norm['train']    
    year_months = np.array(ds_helper.year_month['train'])  # ensure this is a numpy array
    assert np.all(np.sort(year_months) == year_months), 'Samples should be sorted by year-month'    
    n_buckets = len(set(y))
    
    uniq_year_months = sorted(set(year_months))
    idx_bins = [int(x) for x in np.linspace(0, len(uniq_year_months), num=n_folds+1)]
    bins = np.array([uniq_year_months[idx] for idx in idx_bins[:-1]], dtype=int)
    idx_fold = np.digitize(year_months, bins) - 1

    df_ym = pd.DataFrame(year_months, columns=['year_month'])
    scores = []
    models = []
    for fold in range(n_folds):
        model = model_handle(**kwargs)
        group = df_ym.loc[idx_fold != fold].reset_index().groupby('year_month').count().to_numpy()
        eval_yms = sorted(set(year_months[idx_fold == fold]))
        eval_set = []
        eval_group = []
        for ym in eval_yms:
            eval_set.append((X[year_months == ym,:], y[year_months == ym]))
            eval_group.append([eval_set[-1][1].size])
        Xtrain = X[idx_fold != fold,:]
        ytrain = y[idx_fold != fold]
        model.fit(Xtrain, ytrain, eval_at=[100],
                  group=group, eval_group=eval_group,
                  feature_name=ds_helper.feature_names,
                  eval_set=eval_set)

        fold_scores = []
        for z in eval_set:
            Xtest, ytest = z
            y_score = model.predict(Xtest)
            df = pd.DataFrame(np.vstack([y_score, ytest]).T, columns=['pred', 'true']).sort_values('pred', ascending=False)            
            fold_scores.append(ndcg(df.true.values, df.pred.values, k=100, form='exp'))
        scores.append(fold_scores)
        models.append(model)

    return scores, models


def get_cv_grid_args(params_grid):
    params_list = [dict()]
    for key, vals in params_grid.items():
        new_params_list = []
        for params in params_list:
            for val in vals:
                p = params.copy()
                p[key] = val
                new_params_list.append(p)
        params_list = new_params_list

    return params_list

def calc_cta_momentum_signals(ts_prices):
    """Calculate the intermediate CTA Momentum signals following Baz et al.
    
    This function calculates the 16 intermediate momentum signals, following
    the paper 'Dissecting Investment Strategies in the Cross Section and Time Series'
    by Baz et al, published in 2015.
    https://www.cmegroup.com/education/files/dissecting-investment-strategies-in-the-cross-section-and-time-series.pdf
    """
    S = [8, 16, 32]
    L = [24, 48, 96]

    HL = lambda n : np.log(0.5) / np.log(1 - 1/n)

    signals = []
    signal_names = []
    for j in range(3):
        x = ts_prices.ewm(halflife=HL(S[j])).mean() - ts_prices.ewm(halflife=HL(L[j])).mean()
        y = x / ts_prices.rolling(63).std()
        z = y / y.rolling(252).std()
        u = x * np.exp(-np.power(x, 2) / 4) / 0.89    
        signals.extend([x, y, z, u])
        signal_names.extend([n + str(j) for n in ['x', 'y', 'z', 'u']])

    df_signals_daily = pd.concat(signals, axis=1)
    df_signals_daily.index = pd.DatetimeIndex(df_signals_daily.index)
    df_signals_daily.columns = signal_names
    return df_signals_daily.resample('M').last()

def get_performance(ds_helper, model, n_stocks=100, dataset='validation', weighting='equal'):
    year_month_vals = ds_helper.year_month[dataset]
    
    prices_ts = ds_helper.featureObj._time_series['daily_prices']
    full_period_rtns = []
    model_period_rtns = []
    sorted_year_months = sorted(set(year_month_vals))
    for ym in sorted_year_months:
        idx_ym = (ym == year_month_vals)
        y_score = model.predict(ds_helper.X[dataset][idx_ym,:])
        idx_score = np.argsort(y_score)[-n_stocks:]
        idx_symbol = ds_helper.featureObj.symbols[ds_helper.symbol[dataset][idx_ym][idx_score]]
        
        period_end = pd.Timestamp(datetime.datetime.strptime(str(ym), '%Y%m')) + pd.tseries.offsets.MonthEnd(0)
        period_start = period_end - pd.Timedelta(91, unit='d')
        idx_period = (period_start <= prices_ts.index) & (prices_ts.index <= period_end)
        period_prices = prices_ts.loc[idx_period, idx_symbol]
        rtns = np.log(np.maximum(period_prices, ds_helper.featureObj.price_tol) / \
                      np.maximum(period_prices.shift(1), ds_helper.featureObj.price_tol))
        rtns = rtns.iloc[1:,:]
        rtns.fillna(0, inplace=True)
        #print((rtns.std() * np.sqrt(252)).sort_values()[:5])
        
        if weighting == 'equal':
            weights = np.ones((idx_score.size,)) / idx_score.size
        elif weighting == 'hrp':        
            hrp = HRPOpt(rtns)
            hrp.optimize()
            weights = pd.Series(hrp.clean_weights())
        elif weighting == 'minvar':
            weights, _ = optim.optimize_min_variance(np.cov(rtns.T))
        else:
            raise ValueError(f'Unsupported weighting scheme: {weighting}')

        assert np.isclose(weights.sum(), 1.0, atol=0.01), 'Weights must sum to 1.0'
        weights /= weights.sum()
        true_rtns = ds_helper.y_reg[dataset][idx_ym]
        model_period_rtns.append((weights * true_rtns[idx_score]).sum())
        assert sorted_year_months.index(ym) != 9
    return np.array(model_period_rtns)
