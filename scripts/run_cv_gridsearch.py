"""Script to quickly download time series data using multiple threads."""

import argparse
import json
import lightgbm as lgb
import numpy as np
import os
import sys
import time

from eodfundeq import StockFeatureAnalyzer, DatasetHelper
from eodfundeq import utils

from eodhistdata import EODHelper
from eodhistdata.private_constants import API_TOKEN, BASE_PATH


def main(argv):
    t0 = time.time()
    output_file = os.path.join(BASE_PATH, 'gridsearch.json')

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default='1999-12-31',
                        help="First date of data to download")
    parser.add_argument("--end", type=str, default='2022-12-31',
                        help="Last date of data to download")                        
    args = parser.parse_args()

    eod_helper = EODHelper(
        api_token=API_TOKEN, base_path=BASE_PATH)
    symbols = eod_helper.get_non_excluded_exchange_symbols('US')

    featureObj = StockFeatureAnalyzer(API_TOKEN, BASE_PATH, symbols=symbols,
                                    start=args.start, end=args.end)
    featureObj.load_time_series()

    momentum_windows = [3, 6, 12]

    mom_features = dict()
    for window in momentum_windows:
        mom_array = featureObj.get_momentum(window)
        mom_features[f'mom_{window}m'] = mom_array
        #mom_features[f'mom_{window}m_p'] = utils.calc_feature_percentiles(mom_array)
        
        vol = featureObj.get_volatility(window)[f'volatility_{window}m']
        mom_array_norm = mom_array / np.clip(vol, 0.03, np.inf)
        mom_features[f'mom_{window}m_norm'] = mom_array_norm

    cta_mom_features = featureObj.get_cta_momentum_signals()

    mf = mom_features | cta_mom_features
    ds_helper = DatasetHelper(featureObj, mf)
    ds_helper.return_window = 1
    ds_helper.filter_max_monthly_volume = np.inf
    ds_helper.exclude_nan = True
    ds_helper.n_months_valid = 36
    ds_helper.n_months_test = 24

    params_grid = dict(
        learning_rate=[1e-4, 1e-3, 1e-2, 1e-1, 1],
        max_depth = [2, 4, 6, 8, 10],
        n_estimators =  [20, 40, 80, 160, 320],
        feature_fraction = [0.6, 0.8, 1.0],
        label_gain=[[(2 ** w) - 1.0 for w in range(featureObj.n_buckets)]],
        boosting = ['gbdt', 'dart'],  # 'gbdt' is default    
    )

    grid_kwargs = utils.get_cv_grid_args(params_grid)

    for j, gkwargs in enumerate(grid_kwargs):
        print(f'Iteration {j}')
        scores, _ = utils.cross_validate_gbm(
            lgb.LGBMRanker, ds_helper, n_folds=5, **gkwargs)

        # Read existing params, add result, and re-write to file
        if os.path.isfile(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
        else:
            results = []
        results.append((j, scores, gkwargs))
        with open(output_file, "w") as f:
            f.write(json.dumps(results))

    print('======================================================')                
    print(f'Download Complete. Elapsed time = {time.time() - t0}')


if __name__ == "__main__":
    main(sys.argv[1:])
