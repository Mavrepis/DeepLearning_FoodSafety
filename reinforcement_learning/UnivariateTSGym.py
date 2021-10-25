import numpy as np
from sklearn.metrics import mean_absolute_error


def r_score(y_true, y_pred):
    return 1/np.log(mean_absolute_error(y_true, y_pred)+1.1)


class Observation(object):
    def __init__(self, state, target):
        self.target = target
        self.state = state


class Environment(object):
    def __init__(self, dataframe = None):
        self.dataframe = dataframe

        train_start = 0
        train_end = len(dataframe)
        self.history_window = 10
        self.output_lenght = 1

        result = []
        data = dataframe['recalls']
        self.unique_timestamp = data.reset_index(drop=True).index

        for index in range(train_start, train_end - (self.history_window + self.output_lenght)):
            result.append(data[index: index + self.history_window + self.output_lenght])

        result = np.array(result)  # shape (samples, sequence_length)
        all_train = result[train_start:train_end, :]

        self.train = all_train[:, : -self.output_lenght]
        self.y_true = all_train[:, -self.output_lenght:]
        self.timestamp = 0
        self.n = len(self.unique_timestamp)
        self.unique_idx = 0

        # Needed to compute final score
        self.full_y_true = []
        self.full_y_pred = []

    def reset(self):
        self.unique_idx = 0
        state = self.train[self.unique_idx]
        target = [0]
        observation = Observation(state, target)
        return observation

    def step(self, target):

        if self.unique_idx == (self.n-self.history_window-2):
            done = True
            observation = None
            reward = r_score([self.y_true[self.unique_idx]], [target])

            self.full_y_pred.append(target)
            self.full_y_true.append(self.y_true[self.unique_idx])

            # print(reward)
            score = r_score(self.full_y_true, self.full_y_pred)
            info = {'final_score': score}
        else:
            # print('Target in env = ' + str(target))
            reward = r_score([self.y_true[self.unique_idx]], [target])
            self.full_y_pred.append(target)
            self.full_y_true.append(self.y_true[self.unique_idx])
            done = False
            info = {}
            self.unique_idx += 1
            state = self.train[self.unique_idx]
            target=[0]

            observation = Observation(state, target)

        return observation, reward, done, info

    def __str__(self):
        return "Environment()"


def make(data):
    return Environment(dataframe=data)
