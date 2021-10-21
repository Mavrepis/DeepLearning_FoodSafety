"""
Implementation of "A Deep Learning Approach using Natural Language Processing
and Time-series Forecasting towards enhanced Food Safety". This code is responsible for
all the experiments that had to do with the GluonTS models used for the comparison in the paper.
"""
import os
# https://gluon-ts.mxnet.io/examples/basic_forecasting_tutorial/tutorial.html
from pathlib import Path
import argparse
import pandas as pd
import mxnet.context as ctx
import utils as ut
import gluonts_helpers as gh
from tqdm import tqdm


def parse_arguments():
    """ Parses program's arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('./data'))
    parser.add_argument('--window_steps', type=int, default=2)
    parser.add_argument('--surrogation_factor', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--use_gpu', type=int, default=0)
    parser = parser.parse_args()
    return parser


if __name__ == '__main__':
    args = parse_arguments()
    datasets = args.data_path
    surrogation_factor = args.surrogation_factor
    NUMBER_OF_SURROGATE = 100

    for file in os.listdir(datasets):
        if file.endswith('.csv'):
            df = pd.read_csv(datasets / file, parse_dates=['key_as_string', 'priceStringDate'],
                             index_col='priceStringDate')
            df = ut.preprocess_dataframe(data=df)
            df = ut.create_surrogate(df, surrogation_factor, NUMBER_OF_SURROGATE)
        else:
            continue

        custom_dataset = df
        custom_ds_metadata = {'prediction_length': 4,
                              'context_length': 12,
                              'freq': '1M'}

        NUMBER_OF_TEST = args.window_steps
        ctx = ctx.cpu() if not args.use_gpu else ctx.gpu()

        for test_number in tqdm(range(NUMBER_OF_TEST)):
            # Create formatted training/testing data for GluonTS
            train_ds, test_ds = gh.create_gluonts_data(custom_dataset, test_number, NUMBER_OF_TEST,
                                                       custom_ds_metadata['prediction_length'],
                                                       custom_ds_metadata['freq'])

            estimators = gh.create_estimators(pred_len=custom_ds_metadata['prediction_length'],
                                              ctx_len=custom_ds_metadata['context_length'],
                                              freq=custom_ds_metadata['freq'],
                                              ctx=ctx,
                                              epochs=args.epochs)
            predictors = gh.train_estimators(estimators, train_ds)

            predictor_names = ["Deep AR Estimator", "Simple Feed Forward Estimator", "Deep Factor Estimator",
                               "MQ-CNN Estimator", "WaveNet Estimator", "Naive Seasonal Estimator"]

            predictors_dict = {k: v for k, v in zip(predictor_names, predictors)}
            forecasts, timeseries = gh.make_forecasts(predictors, test_ds, 100)
            results = gh.generate_evaluation(forecasts, timeseries, test_ds)
            ut.write_results(results, surrogation_factor, test_number)

        # Produce a sample plot
        gh.plot_prob_forecasts(timeseries[0], forecasts[0][0], ts_name="Sample timeseries",
                               predictor_name="Sample predictor")
