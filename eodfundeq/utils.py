import numpy as np
import pandas as pd
import warnings

from eodfundeq.constants import ReturnTypes


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

def calc_feature_percentiles(array, axis=1):
    """Calculate the percentiles across rows for an array of features."""
    order = array.argsort()
    ranks = order.argsort().astype(np.float32)
    ranks[np.isnan(array)] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        percentiles = ranks / np.nanmax(ranks, axis=axis, keepdims=True)
    return percentiles

def bucket_features(array, n_buckets, axis=1):
    """Method to bucket features by their rank, according to the specified axis"""
    percentiles = calc_feature_percentiles(array, axis=axis)
    edges = np.linspace(0, 1, n_buckets+1)[:-1]
    bucketed = -1 + np.digitize(percentiles, edges)

    # Make sure the bucketed values are -1 if the value is NaN
    bucketed[np.isnan(array)] = -1
    return bucketed

def calc_ndcg(y_true, y_score, k=None, form='exp'):
    df = pd.DataFrame(np.vstack([y_score.flatten(), y_true.flatten()]).T,
                      columns=['score', 'true'])
    df.sort_values('score', ascending=False, inplace=True)
    if k is None:
        k = df.shape[0]
    k = min(k, df.shape[0])
    relevance = df.true.values[:k]
    ideal_rel = df.true.sort_values(ascending=False).iloc[:k]
    discount = np.log2(np.arange(1, k+1) + 1)
    if form in ('exp', 'exponential'):
        DCG = (2 ** relevance - 1) / discount
        IDCG = (2 ** ideal_rel - 1) / discount
    else:
        DCG = relevance / discount
        IDCG = ideal_rel / discount

    return DCG.sum() / IDCG.sum()    

def cross_validate_gbm_cls(model_handle, ds_helper, n_folds=5, k=100, direction='bull', **kwargs):
    """Perform cross-validatation for gbm."""
    n_buckets = ds_helper.featureObj.n_buckets
    X = ds_helper.X['train']
    y_rtn = ds_helper.y_reg['train']
    if direction == 'bull':
        y = ds_helper.y_class_norm['train']
    elif direction == 'bear':
        y = n_buckets - ds_helper.y_class_norm['train'] - 1
    else:
        raise ValueError('Direction must be one of "bull" or "bear".')
    year_months = np.array(ds_helper.year_month['train'])  # ensure this is a numpy array
    assert np.all(np.sort(year_months) == year_months), 'Samples should be sorted by year-month'

    uniq_year_months = sorted(set(year_months))
    idx_bins = [int(x) for x in np.linspace(0, len(uniq_year_months), num=n_folds+1)]
    bins = np.array([uniq_year_months[idx] for idx in idx_bins[:-1]], dtype=int)
    idx_fold = np.digitize(year_months, bins) - 1

    df_ym = pd.DataFrame(year_months, columns=['year_month'])
    scores = []
    rtns = []
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
        model.fit(Xtrain, ytrain, eval_at=[k],
                  group=group, eval_group=eval_group,
                  feature_name=ds_helper.feature_names,
                  eval_set=eval_set)

        fold_scores = []
        fold_rtns = []
        for ym in eval_yms:
            Xtest = X[year_months == ym,:]
            ytest = y[year_months == ym]
            ytest_rtn = y_rtn[year_months == ym]
            y_score = model.predict(Xtest)
            df = pd.DataFrame(np.vstack([y_score, ytest, ytest_rtn]).T, 
                              columns=['pred', 'true', 'rtn']).sort_values('pred', ascending=False)
            fold_scores.append(calc_ndcg(df.true.values, df.pred.values, k=k, form='exp'))
            fold_rtns.append(df.rtn.values[:k].mean())
        scores.append(fold_scores)
        rtns.append(fold_rtns)
        models.append(model)
    return scores, rtns, models

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

def get_date_offset(frequency):
    """Get the date offset appropriate for a particular frequency."""
    if frequency == 'd':
        offset = pd.tseries.offsets.Day()
    elif frequency == 'b':
        offset = pd.tseries.offsets.BusinessDay()
    elif frequency == 'w':
        offset = pd.tseries.offsets.Week()
    elif frequency == 'm':
        offset = pd.tseries.offsets.MonthEnd()
    elif frequency == 'y':
        offset = pd.tseries.offsets.YearEnd()
    else:
        raise NotImplementedError(f'Unsupported frequency: {frequency}')
    return offset

def convert_column_to_panel(value, index, column):
    """Convert column data into a time series panel."""
    ts_long = pd.concat([pd.Series(index, dtype='datetime64[ns]'),
                         pd.Series(value, dtype='float'),
                         pd.Series(column, dtype='str')], axis=1)
    ts_long.columns=['index', 'value', 'column']
    return ts_long.pivot(index='index', values='value', columns='column').sort_index()
