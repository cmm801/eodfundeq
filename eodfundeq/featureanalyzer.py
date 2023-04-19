import numpy as np
import pandas as pd

import pyfintools.tools.freq


class AnalyzeFeatures(object):
    def __init__(self, featureObj, clip=None):
        self.featureObj = featureObj
        self.clip = clip if clip is not None else (-np.inf, np.inf)

    @property
    def n_buckets(self):
        return self.featureObj.n_buckets

    def bucket_results(self, metric_vals, return_window, clip=None, frequency='m'):
        if clip is None:
            clip = self.clip
        return_vals = np.clip(self.featureObj.get_future_returns(return_window), *clip)
        assert metric_vals.shape == return_vals.shape, 'Shape of metric values must align with returns'

        bucketed_rtns = []
        bucketed_nobs = []
        high_vals = []
        low_vals = []

        all_years = sorted(set([x.year for x in self.featureObj.dates[frequency]]))
        annual_high_vals = {y: [] for y in all_years}
        annual_low_vals = {y: [] for y in all_years}
        for idx_date, date in enumerate(self.featureObj.dates[frequency]):
            idx_keep = ~np.isnan(metric_vals[idx_date,:]) & \
                       ~np.isnan(return_vals[idx_date,:]) & \
                       self.featureObj.good_mask[idx_date,:]

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
        for idx_date, date in enumerate(self.featureObj.dates[frequency]):
            idx_keep = ~np.isnan(metric_vals[idx_date,:]) & \
                       self.featureObj.good_mask[idx_date,:]
            metric_row = metric_vals[idx_date, idx_keep]
            if not metric_row.size:
                continue
            bins = np.quantile(metric_row, np.linspace(0, 1, self.n_buckets+1))
            bins[-1] += .01  # Hack to avoid max value being assigned to its own bucket
            buckets[idx_date, idx_keep] = np.digitize(metric_row, bins, right=False)
        return buckets

    def get_performance_ts(self, metric_vals, return_window, clip=None, frequency='m'):
        period_rtns, num_obs, ann_tstat_ts, overall_tstat = self.bucket_results(
            metric_vals, return_window=return_window, clip=clip)
        idx_keep_rows = np.min(num_obs, axis=1) >= self.featureObj.filter_min_obs
        period_rtns = period_rtns[idx_keep_rows, :]
        good_dates = self.featureObj.dates[frequency][idx_keep_rows]

        if period_rtns.shape[0] == 0:
            return pd.DataFrame(), pd.DataFrame(), ann_tstat_ts, np.nan

        # Convert to approximate monthly returns so we can compute hypothetical performance
        monthly_rtns = -1 + (1 + period_rtns) ** (1/return_window)
        perf_ts = pd.DataFrame(np.cumprod(1 + monthly_rtns, axis=0), index=good_dates)
        obs_ts = pd.DataFrame(num_obs[idx_keep_rows], index=good_dates)
        return perf_ts, obs_ts, ann_tstat_ts, overall_tstat

    def get_bucketed_returns_summary(self, metric_vals, return_windows: list, 
                                     clip: Optional[tuple] = None, frequency: str = 'm'):
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
            periods_per_year = int(pyfintools.tools.freq.get_periods_per_year(frequency))
            n_years = perf_ts.shape[0] / periods_per_year
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
                    mom_ts = self.featureObj.get_momentum(momentum_window, lag=lag)
                    key = f'mom_{momentum_window}m'
                    if lag > 0:
                        key += str(lag)
                    results[key] = self.get_bucketed_returns_summary(
                        mom_ts, return_windows=return_windows, clip=clip)
        return results

    def get_bucket_summary_for_fundamental_ratios(self, return_windows, fillna=False,
                                                  n_periods=None, min_obs=4, clip=None):
        self.featureObj.calc_all_fundamental_ratios(fillna=fillna,
            n_periods=n_periods, min_obs=min_obs)

        results = dict()
        for ratio_type in FundamentalRatios:
            results[ratio_type.value] = self.get_bucketed_returns_summary(
                self.featureObj.fundamental_ratios[ratio_type.value],
                return_windows=return_windows,
                clip=clip)
        return results
