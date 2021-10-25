import json

import numpy as np
import pandas as pd
import math
from utils import make_model, preprocess_dataframe

import random
from collections import deque

from UnivariateTSGym import make

CONFIG = 'nothreshold'
ACTIONS = 20  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 50  # timesteps to observe before training
EXPLORE = 50.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
INITIAL_EPSILON = 0.8  # starting value of epsilon
REPLAY_MEMORY = 100  # number of previous transitions to remember
BATCH = 20  # size of minibatch

eps_decay = 10000

EPISODES = 2000

if __name__ == '__main__':
    model = make_model(ACTIONS)
    print("Trained model: 3")

    data = pd.read_csv('./data/confectionery.csv', parse_dates=['key_as_string', 'priceStringDate'],
                       index_col='priceStringDate')
    data = preprocess_dataframe(data)

    env = make(data)
    o = env.reset()

    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 0
    o, r_0, terminal, info = env.step(do_nothing[0])

    s_t = o.state

    OBSERVE = OBSERVATION
    t = 0
    DROP = 10

    epsilon = INITIAL_EPSILON

    for epeisode in range(EPISODES):
        o = env.reset()

        while True:
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = np.zeros([ACTIONS])

            # choose an action epsilon greedy
            # policy = np.ones(ACTIONS) * self.epsilon / ACTIONS
            # best_a = np.argmax(self.Q[state])
            # policy[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
            # np.random.choice(np.arange(self.nA), p=policy) if state in self.Q else random.choice(np.arange(self.nA))
            #
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(np.array(s_t).reshape(1, -1))  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

            # We reduced the epsilon gradually
            # if epsilon > FINAL_EPSILON and t > OBSERVE:
            #     epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            epsilon = FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * math.exp(-1. * t / eps_decay)

            # run the selected action and observed next state and reward
            o, r_t, terminal, info = env.step(action_index)

            if terminal:
                break

            s_t1 = o.state

            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 80, 80, 4
                inputs = np.zeros((BATCH, len(s_t)))  # 32, 80, 80, 4
                targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

                # Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]  # This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    # if terminated, only equals reward

                    inputs[i:i + 1] = state_t  # I saved down s_t
                    targets[i] = model.predict(np.array(state_t).reshape(1, -1))  # Hitting each buttom probability
                    Q_sa = model.predict(np.array(state_t1).reshape(1, -1))

                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        alpha = np.max(Q_sa)
                        targets[i, action_t] = reward_t + GAMMA * alpha
                        # print(targets)

                    # print(model.train_on_batch(inputs, targets))

                    # loss += model.train_on_batch(inputs, targets)[0]
                loss = model.train_on_batch(inputs, targets)[0]

            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 100 == 0:
                print("Now we save model")
                model.save_weights("./model/model.h5", overwrite=True)
                with open("./model/model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif OBSERVE < t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, \
                  "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                  "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")

    print('---THE END----')
