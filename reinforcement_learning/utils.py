import argparse
from pathlib import Path
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense


def parse_arguments():
    """ Parses program's arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=Path, default=Path('./agents/'))
    parser.add_argument('--data_path', type=Path, default=Path('./data/'))
    parser = parser.parse_args()
    return parser


def preprocess_dataframe(data):
    """Helper method to preprocess the dataframe.
       Creates new columns for year,month,recalls and percentage change.
       Limits the date range for the experiment (these data are trustworthy)."""
    data['recalls'] = data['doc_count'] + 1
    data.drop(columns=['product', 'Unnamed: 0', 'key', 'key_as_string', 'doc_count'], inplace=True)
    data = data.resample("M").sum()
    mask = (data.index > '2007-05-31') & (data.index < '2019-09-30')
    data = data.loc[mask]
    data['pct'] = data['recalls'].pct_change()
    return data


def evaluate(preds, true):
    yield mean_squared_error(preds, true)


def make_model(num_actions):
    optimizer = 'adam'
    # batch = 2
    # epoch = 5
    model = Sequential()
    model.add(Dense(32, activation='tanh', input_dim=10))
    model.add(Dense(num_actions, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model