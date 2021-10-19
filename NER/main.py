import pandas as pd
import utils as ut
from training import train_model
from pathlib import Path
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def parse_arguments():
    """ Parses program's arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--demonstration', type=int, default=1)
    parser.add_argument('--model_path', type=Path, default=Path('./model'))
    parser.add_argument('--data_path', type=Path, default=Path('./data/tc_dataset_processed_annotated'))
    parser.add_argument('--epochs', type=int, default=100)
    parser = parser.parse_args()
    return parser


if __name__ == '__main__':
    args = parse_arguments()
    if args.demonstration:
        # For demonstration purposes we use dummy data and the same training/testing.
        TRAIN_DATA = [
            ('Who is Nishanth?', {
                'entities': [(7, 15, 'PERSON')]
            }),
            ('Who is Kamal Khumar?', {
                'entities': [(7, 19, 'PERSON')]
            }),
            ('I like London and Berlin.', {
                'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
            })
        ]

        TRAIN_DICT = {'Who is Nishanth?': {
            'entities': [(7, 15, 'PERSON')]
        },
            'Who is Kamal Khumar?': {
                'entities': [(7, 19, 'PERSON')]
            },
            'I like London and Berlin.': {
                'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
            }
        }
        nlp = train_model(TRAIN_DATA, 100)
        nlp.to_disk(args.model_path)
        result = ut.eval_model(args.model_path, TRAIN_DICT, TRAIN_DATA)
        print(result)
    else:
        # Read data
        data = pd.read_csv(args.data_path)
        # Split data on whether they had a product in the recall or not.
        X = [i for i in range(0, len(data))]
        Y = [1 if row['annotation'] != str((0, 0, '')) else 0 for index, row in data.iterrows()]

        X = np.asarray(X)
        Y = np.asarray(Y)

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        sss.get_n_splits(X, Y)

        for train_index, test_index in sss.split(X, Y):
            # Create data in the required format
            TRAIN_DATA = ut.set_data_list(data, train_index)
            TEST_DICT = ut.set_data_dict(data, test_index)
            TEST_DATA = ut.set_data_list(data, test_index)
            # Train Model
            nlp = train_model(TRAIN_DATA, 100)
            # Save Model
            nlp.to_disk(args.model_path)
            # Evaluate Model
            result = eval_model(TEST_DICT, TEST_DATA)
            print(result)
