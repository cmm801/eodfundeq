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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default='1999-12-31',
                        help="First date of data to download")
    parser.add_argument("--end", type=str, default='2022-12-31',
                        help="Last date of data to download")
    parser.add_argument("--direction", type=str, default='bull',
                        help="Whether to run LTR for 'bull' or 'bear' forecasts.")
    args = parser.parse_args()

    assert args.direction in ('bull', 'bear'), 'Argument "direction" must be "bull" or "bear".'
    output_file = os.path.join(BASE_PATH, f'gridsearch_{args.direction}.json')

    eod_helper = EODHelper(
        api_token=API_TOKEN, base_path=BASE_PATH)
    symbols = eod_helper.get_non_excluded_exchange_symbols('US')

    featureObj = StockFeatureAnalyzer(API_TOKEN, BASE_PATH, symbols=symbols,
                                    start=args.start, end=args.end)
    featureObj.load_ohlcv_data()

    momentum_windows = [3, 6, 12]

    mom_features = dict()
    for window in momentum_windows:
        mom_array = featureObj.get_momentum(window)
        mom_features[f'mom_{window}m'] = mom_array

        vol = featureObj.get_volatility(window * 21)
        mom_array_norm = mom_array / np.clip(vol, 0.03, np.inf)
        mom_features[f'mom_{window}m_norm'] = mom_array_norm

    cta_mom_features = featureObj.get_cta_momentum_signals()

    mf = mom_features | cta_mom_features
    ds_helper = DatasetHelper(featureObj, mf, return_window=1, norm=True)
    ds_helper.exclude_nan = True
    ds_helper.n_months_valid = 60

    params_grid = dict(
        learning_rate=[0.01, 0.1, 1.0],
        max_depth = [2, 4, 6],
        n_estimators =  [80, 160, 240],
        feature_fraction = [0.7, 0.85, 1.0],
        label_gain=[[(2 ** w) - 1.0 for w in range(featureObj.n_buckets)]],
        boosting = ['gbdt'],  # 'gbdt' is default    
    )

    grid_kwargs = utils.get_cv_grid_args(params_grid)

    for j, gkwargs in enumerate(grid_kwargs):
        print(f'Iteration {j}')
        scores, rtns, _ = utils.cross_validate_gbm(
            lgb.LGBMRanker, ds_helper, n_folds=5, direction=args.direction, **gkwargs)

        # Read existing params, add result, and re-write to file
        if os.path.isfile(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
        else:
            results = []
        results.append(dict(iteration=j, score=scores, returns=rtns, args=gkwargs))
        with open(output_file, "w") as f:
            f.write(json.dumps(results))

    print('======================================================')                
    print(f'Calculation complete. Elapsed time = {time.time() - t0}')


if __name__ == "__main__":
    main(sys.argv[1:])
