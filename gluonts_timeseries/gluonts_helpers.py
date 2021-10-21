from gluonts.dataset import common
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.mx.trainer import Trainer
from matplotlib import pyplot as plt


def create_estimators(pred_len, ctx_len, freq, ctx, epochs):
    """ Creates GluonTS estimators for comparison.
        All estimators share the parameters of training hardware and epochs,"""
    estimator_dare = DeepAREstimator(
        prediction_length=pred_len,
        context_length=ctx_len,
        freq=freq,
        trainer=Trainer(epochs=epochs,
                        ctx=ctx),
        use_feat_dynamic_real=False,
        num_parallel_samples=1000
    )
    estimator_sffe = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=pred_len,
        context_length=ctx_len,
        freq=freq,
        trainer=Trainer(ctx=ctx, epochs=epochs)
    )
    estimator_factor = DeepFactorEstimator(
        prediction_length=pred_len,
        context_length=ctx_len,
        freq=freq,
        num_hidden_local=8,
        trainer=Trainer(ctx=ctx,
                        epochs=epochs)
    )
    estimator_MQCNN = MQCNNEstimator(
        prediction_length=pred_len,
        context_length=ctx_len,
        freq=freq,
        trainer=Trainer(ctx=ctx,
                        epochs=epochs,
                        hybridize=False)
    )
    estimator_wave = WaveNetEstimator(
        freq=freq,
        prediction_length=pred_len,
        trainer=Trainer(epochs=epochs, ctx=ctx)
    )
    estimator_seasonal_naive = SeasonalNaivePredictor(
        prediction_length=pred_len,
        freq=freq,
    )
    return [estimator_dare, estimator_sffe, estimator_factor, estimator_MQCNN, estimator_wave, estimator_seasonal_naive]


def create_gluonts_data(df, test_idx, num_tests, pred_length, freq):
    """ GluonTS requires the data in a specific format described in
        https://ts.gluon.ai/tutorials/forecasting/quick_start_tutorial.html#Custom-datasets
        On top of that we implemented the moving window procedure described in the manuscript.
    """

    train_ds = common.ListDataset(
        [{'target': df.pct[
                    test_idx:-num_tests + test_idx - pred_length],
          'start': df.index[test_idx], 'feat_dynamic_real': [
                df.month[test_idx:-num_tests + test_idx - pred_length]]}
         ],
        freq=freq)

    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = common.ListDataset(
        [{'target': df.pct[test_idx:-num_tests + test_idx],
          'start': df.index[test_idx],
          'feat_dynamic_real': [df.month[test_idx:-num_tests + test_idx]]}
         ],
        freq=freq)
    return train_ds, test_ds


def train_estimators(estimators, train_ds):
    """ This method is responsible for training all estimators of the experiment.
        A workaround had to be implemented since SeasonalNaivePredictor does not
        have .train method."""
    trained_estimators = [estimator.train(train_ds) for estimator in estimators[:-1]]
    trained_estimators.append(estimators[-1])
    return trained_estimators


def make_forecasts(predictors, test_data, n_sampl):
    """Takes a dictionary of predictors with their respective name,gluonTS test data,
     number of samples for probabilistic predictions and returns forecasts for each of them.

     timeseries is a list containing all different timeseries of the dataset.

     The entries in the forecast list are a bit more complex.
     They are objects that contain all the sample paths in the form of numpy.ndarray
     with dimension (num_samples, prediction_length),
     the start date of the forecast, the frequency of the time series, etc.
     We can access all these information by simply invoking the corresponding attribute
     of the forecast object.

     """
    forecasts = []
    timeseries = []
    for predictor in predictors:
        # forecast_it and ts_it are iterators
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=predictor,
            # number of sample paths we want for evaluation
            num_samples=n_sampl
        )
        forecasts.append((list(forecast_it)))
        timeseries = list(ts_it)
    return forecasts, timeseries


def generate_evaluation(forecasts, timeseries, test_ds):
    """ Using forecasts and the corresponding timeseries produced metrics
    to quantify each model's performance using the Evaluator provided by GluonTS."""
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    metrics = []
    for forecast in forecasts:
        agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecast), num_series=len(test_ds))
        metrics.append((agg_metrics, item_metrics))
    return metrics


def plot_prob_forecasts(ts_entry, forecast_entry, ts_name='N/A', predictor_name='N/A'):
    """ Custom plotting function used to produce figures for the manuscript."""
    plot_length = 16
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # plt.figure(figsize=(20, 10))
    ts_entry[-plot_length:].plot()  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.title(predictor_name + " on " + ts_name)
    plt.savefig("./figures/" + ts_name + ' - ' + predictor_name + '.png')
    plt.close()
