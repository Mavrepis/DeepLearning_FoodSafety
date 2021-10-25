import os
import re

import numpy as np
import pandas as pd
import torch
from utils import *

from dqn_agent import Agent

N_ACTIONS = 20
NUMBER_OF_TEST = 46
PREDICTION_LEN = 4
data = []

if __name__ == '__main__':
    args = parse_arguments()
    current_agent = Agent(state_size=10, action_size=N_ACTIONS, seed=0)
    for file in os.listdir(args.data_path):
        if file.endswith('.csv'):
            data = pd.read_csv(args.data_path / file,
                               parse_dates=['key_as_string', 'priceStringDate'],
                               index_col='priceStringDate')
            data = preprocess_dataframe(data)
            data = data['pct'].to_numpy()
            product_name = str(file).replace("_", " ")[:-4]
            if Path.exists(args.agents / product_name):
                Agent_path = Path(args.agents / product_name)
                avg_queue = []
                for agent_checkpoint in os.listdir(Agent_path):
                    # Get agent's number (os.listdir not sorted)
                    agent_num = int(re.findall(r'\d+', agent_checkpoint)[0])
                    # Get agent's corresponding data (to match training)
                    initial_index = agent_num - NUMBER_OF_TEST - PREDICTION_LEN
                    agent_data = data[agent_num:initial_index]
                    # Recreate agent's actions
                    action_value_map = np.linspace(np.min(agent_data), np.max(agent_data), N_ACTIONS)
                    # Load pre-trained weights
                    current_agent.qnetwork_local.load_state_dict(torch.load(Agent_path / agent_checkpoint))
                    # Set initial state ( end - state_size : end )
                    state = data[initial_index - current_agent.state_size: initial_index]
                    predictions = []
                    # Get real values
                    true_values = data[initial_index:initial_index + 4]
                    # We predict 4 steps ahead
                    for i in range(4):
                        prediction = action_value_map[current_agent.act(state)]
                        # Update state, append predicted value
                        new_state = np.append(state[1:], prediction)
                        state = new_state
                        predictions.append(prediction)
                    avg_queue.append(next(evaluate(predictions, true_values)))
                print("Average MSE for " + product_name + " agents: ", np.mean(avg_queue))
